"""
Feature pipeline for the Cyber Guardian agent: the recipe's SecOps tables into
the feature store.

Google's ADK recipe loads six CSVs into BigQuery and its tools run SQL over
them: a duplicate check on incidents by host and alert type in the last 24
hours, an asset lookup by hostname, process events by host and parent process,
network connections by host and destination, threat intel by indicator, and a
playbook by triggering condition. The online feature store is a keyed lookup,
not a query engine, so each table is shaped around the question the agent
asks of it: one row per key, holding everything that question needs.

| Feature group                   | Key                    | Answers                                              |
|---------------------------------|------------------------|------------------------------------------------------|
| `secops_assets`                 | `hostname`             | "who owns this host, how critical is it?"            |
| `secops_host_process_events`    | `hostname`             | "what ran on this host?" (events as JSON, filtered in code) |
| `secops_host_network_connections` | `hostname`           | "what did this host talk to?" (connections as JSON)  |
| `secops_threat_intel`           | `ioc_value`            | "is this indicator known bad, and as what?"          |
| `secops_playbooks`              | `triggering_condition` | "what does the playbook say to do?" (steps as JSON)  |
| `secops_incidents`              | `incident_id`          | the incident ledger; the agent appends to it         |
| `secops_incident_index`         | `dedup_key`            | "when was this host last opened for this alert type?" |

The two incident groups are the recipe's one table split by question: the
ledger is written and the index is read, keyed by `host|alert_type`, so the
duplicate check is one lookup instead of a scan.

The CSVs are read from the recipe's repository (set CYBER_GUARDIAN_DATA to a
directory holding them to skip the download).

Run once before starting the agent:

    python feature_pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

import hopsworks
import pandas as pd
from hsfs.feature import Feature

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

FG_VERSION = 1
ASSETS_FG = "secops_assets"
PROCESS_EVENTS_FG = "secops_host_process_events"
NETWORK_FG = "secops_host_network_connections"
THREAT_INTEL_FG = "secops_threat_intel"
PLAYBOOKS_FG = "secops_playbooks"
INCIDENTS_FG = "secops_incidents"
INCIDENT_INDEX_FG = "secops_incident_index"

DATA_URL = "https://raw.githubusercontent.com/google/adk-recipes/main/python/agents/cyber-guardian-agent/sample_data"
DATA_DIR = os.environ.get("CYBER_GUARDIAN_DATA", "sample_data")
FILES = (
    "asset_inventory",
    "endpoint_process_events",
    "incident_management",
    "network_connection_log",
    "response_playbooks",
    "threat_intelligence_kb",
)


def ensure_data(directory: str = DATA_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    for name in FILES:
        path = os.path.join(directory, f"{name}.csv")
        if not os.path.exists(path):
            log.info("Downloading %s.csv → %s", name, path)
            urllib.request.urlretrieve(f"{DATA_URL}/{name}.csv", path)
    return directory


def _text_feature(name: str) -> Feature:
    """A long text column, declared TEXT so it does not count against the online row limit."""
    return Feature(name, type="string", online_type="text")


def _timestamp(series: pd.Series) -> pd.Series:
    """The CSVs' "2025-09-15 08:29:21.456345 UTC" as naive UTC timestamps."""
    return pd.to_datetime(series.str.replace(" UTC", "", regex=False), errors="coerce")


def _grouped_json(frame: pd.DataFrame, key: str, sort_by: str) -> pd.DataFrame:
    """One row per key holding the group's rows as a JSON list, newest first."""
    frame = frame.sort_values(sort_by, ascending=False)
    rows = []
    for value, group in frame.groupby(key, sort=False):
        records = group.drop(columns=[key]).to_dict("records")
        for record in records:
            for column, cell in list(record.items()):
                if isinstance(cell, pd.Timestamp):
                    record[column] = cell.isoformat()
                elif pd.isna(cell):
                    record[column] = None
        rows.append({key: value, "rows": json.dumps(records), "row_count": len(records)})
    return pd.DataFrame(rows)


def read_tables(directory: str) -> dict[str, pd.DataFrame]:
    read = lambda name: pd.read_csv(os.path.join(directory, f"{name}.csv"))  # noqa: E731

    assets = read("asset_inventory").rename(
        columns={
            "Hostname": "hostname", "IPAddress": "ip_address", "OS": "os",
            "AssetType": "asset_type", "Owner": "owner", "BusinessCriticality": "business_criticality",
        }
    ).drop_duplicates("hostname")

    events = read("endpoint_process_events").rename(
        columns={
            "Hostname": "hostname", "Username": "username", "ProcessName": "process_name",
            "ProcessID": "process_id", "ParentProcessName": "parent_process_name",
            "ParentProcessID": "parent_process_id", "CommandLine": "command_line",
            "EventTimestamp": "event_timestamp",
        }
    )
    events["event_timestamp"] = _timestamp(events["event_timestamp"])
    process_events = _grouped_json(events, "hostname", "event_timestamp")

    network = read("network_connection_log").rename(columns={"source_host": "hostname"})
    network["log_timestamp"] = _timestamp(network["log_timestamp"])
    connections = _grouped_json(network, "hostname", "log_timestamp")

    intel = read("threat_intelligence_kb").rename(
        columns={
            "IOC_Value": "ioc_value", "IOC_Type": "ioc_type", "IsMalicious": "is_malicious",
            "ThreatName": "threat_name", "Confidence": "confidence", "LastSeen": "last_seen",
        }
    ).drop_duplicates("ioc_value")
    intel["is_malicious"] = intel["is_malicious"].astype(str).str.lower().eq("true")
    intel["last_seen"] = intel["last_seen"].astype(str)

    playbooks_raw = read("response_playbooks").rename(
        columns={
            "PlaybookID": "playbook_id", "TriggeringCondition": "triggering_condition",
            "ActionStep": "action_step", "ActionCommand": "action_command",
            "TargetDescription": "target_description", "RequiresApproval": "requires_approval",
        }
    )
    playbooks_raw = playbooks_raw[[c for c in playbooks_raw.columns if not c.startswith("bool_field")]]
    playbooks_raw["requires_approval"] = playbooks_raw["requires_approval"].astype(str).str.lower().eq("true")
    playbooks = []
    for condition, group in playbooks_raw.groupby("triggering_condition"):
        steps = group.sort_values("action_step").to_dict("records")
        playbooks.append(
            {
                "triggering_condition": condition,
                "playbook_id": steps[0]["playbook_id"],
                "steps": json.dumps(
                    [
                        {
                            "step": int(s["action_step"]),
                            "action": s["action_command"],
                            "target": s["target_description"],
                            "requires_approval": bool(s["requires_approval"]),
                        }
                        for s in steps
                    ]
                ),
            }
        )
    playbooks = pd.DataFrame(playbooks)

    incidents = read("incident_management").rename(
        columns={
            "IncidentID": "incident_id", "AlertType": "alert_type", "Status": "status",
            "Severity": "severity", "PrimaryHost": "primary_host", "PrimaryUser": "primary_user",
            "Summary": "summary", "CreationTimestamp": "created_at",
        }
    ).drop_duplicates("incident_id")
    incidents["created_at"] = _timestamp(incidents["created_at"])

    latest = incidents.sort_values("created_at").groupby(["primary_host", "alert_type"]).tail(1)
    index = pd.DataFrame(
        {
            "dedup_key": latest["primary_host"] + "|" + latest["alert_type"],
            "incident_id": latest["incident_id"],
            "created_at": latest["created_at"],
        }
    )
    return {
        "assets": assets, "process_events": process_events, "connections": connections,
        "intel": intel, "playbooks": playbooks, "incidents": incidents, "index": index,
    }


def main() -> None:
    tables = read_tables(ensure_data())
    for name, frame in tables.items():
        log.info("  %-15s %d rows", name, len(frame))
    now = pd.Timestamp.utcnow().tz_localize(None)

    project = hopsworks.login()
    fs = project.get_feature_store()

    def create(name, description, key, event_time, features, frame):
        frame = frame.copy()
        if event_time not in frame.columns:
            frame[event_time] = now
        fg = fs.get_or_create_feature_group(
            name=name, version=FG_VERSION, description=description, primary_key=[key],
            event_time=event_time, online_enabled=True, features=features,
        )
        fg.insert(frame, write_options={"wait_for_job": True})
        fs.get_or_create_feature_view(name=name, version=FG_VERSION, query=fg.select_all())
        log.info("Ready: %s", name)

    create(
        ASSETS_FG, "Asset inventory: owner and business criticality per host", "hostname", "loaded_at",
        [Feature("hostname", type="string"), Feature("ip_address", type="string"), Feature("os", type="string"),
         Feature("asset_type", type="string"), Feature("owner", type="string"),
         Feature("business_criticality", type="string"), Feature("loaded_at", type="timestamp")],
        tables["assets"],
    )
    create(
        PROCESS_EVENTS_FG, "Endpoint process events per host, newest first, as JSON", "hostname", "loaded_at",
        [Feature("hostname", type="string"), _text_feature("rows"), Feature("row_count", type="bigint"),
         Feature("loaded_at", type="timestamp")],
        tables["process_events"],
    )
    create(
        NETWORK_FG, "Network connections per source host, newest first, as JSON", "hostname", "loaded_at",
        [Feature("hostname", type="string"), _text_feature("rows"), Feature("row_count", type="bigint"),
         Feature("loaded_at", type="timestamp")],
        tables["connections"],
    )
    create(
        THREAT_INTEL_FG, "Threat intelligence: known indicators and the threat they belong to", "ioc_value", "loaded_at",
        [Feature("ioc_value", type="string"), Feature("ioc_type", type="string"),
         Feature("is_malicious", type="boolean"), Feature("threat_name", type="string"),
         Feature("confidence", type="string"), Feature("last_seen", type="string"),
         Feature("loaded_at", type="timestamp")],
        tables["intel"],
    )
    create(
        PLAYBOOKS_FG, "Response playbooks: the steps for a triggering condition, as JSON", "triggering_condition", "loaded_at",
        [Feature("triggering_condition", type="string"), Feature("playbook_id", type="string"),
         _text_feature("steps"), Feature("loaded_at", type="timestamp")],
        tables["playbooks"],
    )
    create(
        INCIDENTS_FG, "Incident ledger; the agent appends one row per incident it opens", "incident_id", "created_at",
        [Feature("incident_id", type="string"), Feature("alert_type", type="string"), Feature("status", type="string"),
         Feature("severity", type="string"), Feature("primary_host", type="string"),
         Feature("primary_user", type="string"), _text_feature("summary"), Feature("created_at", type="timestamp")],
        tables["incidents"],
    )
    create(
        INCIDENT_INDEX_FG, "The latest incident per host and alert type, for the duplicate check", "dedup_key", "created_at",
        [Feature("dedup_key", type="string"), Feature("incident_id", type="string"),
         Feature("created_at", type="timestamp")],
        tables["index"],
    )
    log.info("Done. The agent can now triage, investigate and respond.")


if __name__ == "__main__":
    main()
