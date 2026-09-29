"""
The SecOps data as the agent's tools see it, over the feature store.

The recipe's tools ran SQL against BigQuery; these run keyed lookups against
the feature groups `feature_pipeline.py` shaped for exactly these questions.
Filtering that the SQL did (by parent process, by destination IP, by age) is
done here in Python on the row the lookup returned.

Two things write: `create_incident` appends to the incident ledger and
updates the duplicate-check index, and `execute_response` simulates a SOAR
action, as the recipe's did. Both are no-ops under evaluation
(`in_evaluation()`), so a sandboxed suite can run against the deployment that
handles real alerts. An incident opened earlier in the same conversation
counts as a duplicate too, so the duplicate path can be seen without a write.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd
from hopsworks_agents.protocol.autoevents import current_context
from hopsworks_agents.protocol.evaluation import in_evaluation
from langchain_core.tools import tool

log = logging.getLogger(__name__)

FG_VERSION = 1
ASSETS_FG = "secops_assets"
PROCESS_EVENTS_FG = "secops_host_process_events"
NETWORK_FG = "secops_host_network_connections"
THREAT_INTEL_FG = "secops_threat_intel"
PLAYBOOKS_FG = "secops_playbooks"
INCIDENTS_FG = "secops_incidents"
INCIDENT_INDEX_FG = "secops_incident_index"
#: the recipe's window for "the same alert again"
DUPLICATE_WINDOW = timedelta(hours=24)
#: the recipe's LIMIT on log queries
MAX_LOG_ROWS = 10
#: incidents opened in this conversation, kept in session memory under this key
SESSION_INCIDENTS_KEY = "incidents_opened"

_project = hopsworks.login()
_fs = _project.get_feature_store()
_views: dict[str, object] = {}


def _view(name: str):
    """A serving-initialised feature view, cached for the process."""
    if name not in _views:
        view = _fs.get_feature_view(name=name, version=FG_VERSION)
        if view is None:
            raise LookupError(
                f"Feature view {name!r} v{FG_VERSION} does not exist — "
                "run cyber_guardian/feature_pipeline.py first"
            )
        view.init_serving()
        _views[name] = view
    return _views[name]


def _clean(record: dict) -> dict | None:
    if all(pd.isna(v) if not isinstance(v, (list, dict)) else False for v in record.values()):
        return None
    return {k: (None if not isinstance(v, (list, dict, str)) and pd.isna(v) else v) for k, v in record.items()}


def _lookup(view_name: str, entry: dict) -> dict | None:
    """One keyed online read, or None when the key is absent."""
    try:
        row = _view(view_name).get_feature_vector(entry, return_type="pandas")
    except Exception:  # noqa: BLE001 — a missing key raises on some versions
        log.exception("Lookup in %s failed for %s", view_name, entry)
        return None
    if row is None or getattr(row, "empty", False):
        return None
    return _clean(row.iloc[0].to_dict())


def _lookup_many(view_name: str, entries: list[dict]) -> list[dict]:
    if not entries:
        return []
    try:
        frame = _view(view_name).get_feature_vectors(entries, return_type="pandas")
    except Exception:  # noqa: BLE001
        log.exception("Batch lookup in %s failed", view_name)
        return []
    if frame is None or getattr(frame, "empty", False):
        return []
    return [r for r in (_clean(row) for row in frame.to_dict("records")) if r]


def _rows(record: dict | None) -> list[dict]:
    try:
        rows = json.loads((record or {}).get("rows") or "[]")
    except ValueError:
        return []
    return rows if isinstance(rows, list) else []


def _ctx():
    return current_context.get(None)


def _session_incidents() -> dict[str, str]:
    """Incidents this conversation opened: dedup_key → incident id."""
    ctx = _ctx()
    if ctx is None or ctx.memory is None:
        return {}
    try:
        return json.loads(ctx.memory.get_state("session", ctx.conversation_id, SESSION_INCIDENTS_KEY) or "{}")
    except ValueError:
        return {}


def _remember_incident(dedup_key: str, incident_id: str) -> None:
    ctx = _ctx()
    if ctx is None or ctx.memory is None:
        return
    opened = _session_incidents()
    opened[dedup_key] = incident_id
    ctx.memory.set_state("session", ctx.conversation_id, SESSION_INCIDENTS_KEY, json.dumps(opened))


# ── the tools ────────────────────────────────────────────────────────────────


@tool
def triage_query(hostname: str, alert_type: str) -> str:
    """Check whether this host already has an open incident of this type in the last 24 hours, and
    enrich the host with its owner and business criticality from the asset inventory.

    Args:
        hostname: The host named in the alert, e.g. "winsrv0221".
        alert_type: The alert's classification: IOC_MATCH, EDR_DETECTION, PHISHING_EMAIL or UNCATEGORIZED.
    """
    hostname = (hostname or "").strip()
    alert_type = (alert_type or "").strip().upper()
    dedup_key = f"{hostname}|{alert_type}"
    existing = _session_incidents().get(dedup_key)
    if existing is None:
        latest = _lookup(INCIDENT_INDEX_FG, {"dedup_key": dedup_key})
        if latest and latest.get("created_at") is not None:
            created = pd.Timestamp(latest["created_at"]).to_pydatetime().replace(tzinfo=None)
            if datetime.now(timezone.utc).replace(tzinfo=None) - created < DUPLICATE_WINDOW:
                existing = str(latest.get("incident_id"))
    if existing:
        return json.dumps({"is_duplicate": True, "existing_incident": existing})
    asset = _lookup(ASSETS_FG, {"hostname": hostname})
    context = (
        {
            "owner": asset.get("owner"),
            "criticality": asset.get("business_criticality"),
            "os": asset.get("os"),
            "asset_type": asset.get("asset_type"),
            "ip_address": asset.get("ip_address"),
        }
        if asset
        else "No asset context found."
    )
    return json.dumps({"is_duplicate": False, "asset_context": context})


@tool
def investigation_query(
    alert_type: str, hostname: str, parent_process: str = "", destination_ip: str = ""
) -> str:
    """Pull the logs an investigation needs for a host: process events and network connections.

    For EDR_DETECTION, the process events under `parent_process` and every outbound connection
    of the host. For IOC_MATCH, the connections to `destination_ip` and the process events around
    them. For PHISHING_EMAIL and anything else, the host's connections and events as they are.
    Newest first, at most 10 of each.

    Args:
        alert_type: IOC_MATCH, EDR_DETECTION, PHISHING_EMAIL or UNCATEGORIZED.
        hostname: The host to investigate.
        parent_process: For EDR alerts, the parent process named in the alert, e.g. "services.exe".
        destination_ip: For IOC matches, the indicator IP the host connected to.
    """
    hostname = (hostname or "").strip()
    alert_type = (alert_type or "").strip().upper()
    events = _rows(_lookup(PROCESS_EVENTS_FG, {"hostname": hostname}))
    connections = _rows(_lookup(NETWORK_FG, {"hostname": hostname}))
    if alert_type == "EDR_DETECTION" and parent_process:
        parent = parent_process.strip().lower()
        matched = [e for e in events if str(e.get("parent_process_name") or "").lower() == parent]
        events = matched or events
    if alert_type == "IOC_MATCH" and destination_ip:
        target = destination_ip.strip()
        connections = [c for c in connections if str(c.get("destination_ip")) == target]
    return json.dumps(
        {
            "hostname": hostname,
            "process_events": events[:MAX_LOG_ROWS],
            "network_connections": connections[:MAX_LOG_ROWS],
        }
    )


@tool
def threat_intel_query(indicators: list[str]) -> str:
    """Look up indicators of compromise (IPs, domains, URLs, file hashes) in the threat
    intelligence knowledge base. Unknown indicators come back as not malicious, threat Unknown.

    Args:
        indicators: The indicator values to look up, as a list even for one.
    """
    values = []
    for raw in indicators or []:
        value = str(raw or "").strip()
        if value and value not in values:
            values.append(value)
    if not values:
        return json.dumps([])
    found = {r["ioc_value"]: r for r in _lookup_many(THREAT_INTEL_FG, [{"ioc_value": v} for v in values])}
    report = []
    for value in values:
        hit = found.get(value)
        if hit:
            report.append(
                {
                    "ioc": value,
                    "ioc_type": hit.get("ioc_type"),
                    "is_malicious": bool(hit.get("is_malicious")),
                    "threat_name": hit.get("threat_name") or "Unknown",
                    "confidence": hit.get("confidence") or "Unknown",
                    "last_seen": hit.get("last_seen"),
                }
            )
        else:
            report.append(
                {"ioc": value, "is_malicious": False, "threat_name": "Unknown", "confidence": "Unknown",
                 "note": "not in the knowledge base"}
            )
    return json.dumps(report)


@tool
def get_playbook(triggering_condition: str) -> str:
    """The response playbook for a condition, as its ordered steps with whether each needs approval.

    Conditions are written as the playbooks name them: "ThreatName = 'LockbitC2'" or
    "AlertType = 'PHISHING_EMAIL'".

    Args:
        triggering_condition: The condition to match, exactly as above.
    """
    playbook = _lookup(PLAYBOOKS_FG, {"triggering_condition": (triggering_condition or "").strip()})
    if not playbook:
        return json.dumps({"playbook_id": None, "steps": [], "note": f"no playbook for {triggering_condition!r}"})
    try:
        steps = json.loads(playbook.get("steps") or "[]")
    except ValueError:
        steps = []
    return json.dumps({"playbook_id": playbook.get("playbook_id"), "steps": steps})


@tool
def execute_response(action: str, target: str) -> str:
    """Execute one response action (block an IP, isolate a host, delete emails) against a target.

    Simulated, as in the recipe: a real deployment would call a SOAR playbook here. Only for
    actions that need no approval, or that an analyst approved.

    Args:
        action: The playbook's action command, e.g. "isolate_host".
        target: What to run it against, e.g. the hostname or the IP.
    """
    if in_evaluation():
        log.info("eval mode: not executing %s on %s", action, target)
    else:
        log.info("Executing response action %r on %r", action, target)
    return json.dumps({"status": "success", "action": action, "target": target, "simulated": True})


@tool
def create_incident(alert_type: str, hostname: str, user: str, severity: str, summary: str = "") -> str:
    """Open an incident for this alert in the incident ledger, so the same alert on the same host
    is a duplicate for the next 24 hours.

    Args:
        alert_type: IOC_MATCH, EDR_DETECTION, PHISHING_EMAIL or UNCATEGORIZED.
        hostname: The primary host.
        user: The primary user.
        severity: Critical, High, Medium or Low.
        summary: One line on what was found.
    """
    hostname = (hostname or "").strip()
    alert_type = (alert_type or "").strip().upper()
    incident_id = f"INC-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    dedup_key = f"{hostname}|{alert_type}"
    _remember_incident(dedup_key, incident_id)
    if in_evaluation():
        log.info("eval mode: not recording incident %s", incident_id)
        return json.dumps({"status": "success", "incident_id": incident_id, "recorded": False})
    try:
        _fs.get_feature_group(INCIDENTS_FG, version=FG_VERSION).insert(
            pd.DataFrame(
                [{"incident_id": incident_id, "alert_type": alert_type, "status": "Triage",
                  "severity": severity or "Medium", "primary_host": hostname, "primary_user": user or "",
                  "summary": summary or "", "created_at": now}]
            ),
            storage="online",
            write_options={"wait_for_job": False},
        )
        _fs.get_feature_group(INCIDENT_INDEX_FG, version=FG_VERSION).insert(
            pd.DataFrame([{"dedup_key": dedup_key, "incident_id": incident_id, "created_at": now}]),
            storage="online",
            write_options={"wait_for_job": False},
        )
    except Exception:  # noqa: BLE001
        log.exception("Could not record incident %s", incident_id)
        return json.dumps({"status": "error", "message": "the incident could not be recorded"})
    return json.dumps({"status": "success", "incident_id": incident_id, "recorded": True})


TOOLS = [triage_query, investigation_query, threat_intel_query, get_playbook, execute_response, create_incident]
