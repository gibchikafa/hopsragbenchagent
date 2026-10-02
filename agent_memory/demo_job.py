"""
Deploy the memory demo and talk to it, in one run.

Made to be a Hopsworks job: the job's log is the demonstration. It deploys
`memory_agent.py`, waits for it to answer, then holds two conversations as the
same person and prints what memory did between them.

    python demo_job.py                       # deploy, start, run the script
    python demo_job.py --skip-deploy         # talk to an agent that is already up
    python demo_job.py --strict              # exit 1 if a tier did not do its job

What the two conversations show, in order:

1. **The buffer.** Something said a turn ago is answered from the turn before,
   with no lookup.
2. **The summary.** The demo folds after six messages, so by the end of the
   first conversation the server holds a rolling summary and says which
   messages it covers.
3. **Durable memory, across conversations.** The second conversation is new --
   a different `conversation_id`, no shared history -- but the same `subject`,
   so what the person said about themselves is still there.

   Who that person is is random per run, and the subject is derived from the
   name with a unique suffix, so every run meets the agent as a stranger. With
   a fixed subject this check would pass on facts an earlier run stored, which
   is a check that cannot fail. `--persona` and `--subject` pin it when
   surviving across runs is the thing you want to show.
4. **Session scope staying put.** The working note from the first conversation
   is *not* in the second. That is the line between "who this person is" and
   "what they happened to be doing", and it is the one worth seeing.
5. **Forgetting.** A fact deleted on request is gone from the next answer.

As a job:

    hopsworks.login().get_jobs_api().create_job(
        "memorydemo", {"type": "pythonJobConfiguration",
                       "appPath": "/Projects/<project>/Resources/demo_job.py",
                       "environmentName": "python-agent-pipeline"})

`memory_agent.py`, `prompts.py` and `requirements.txt` must sit beside this
file for the deploy step; with `--git-url` the deployment pulls them from the
repository instead, and only this script has to be uploaded.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

#: Who the run is about. Random by default, and the subject is derived from it
#: with a unique suffix, so each run meets the agent as a person it has never
#: heard of. That matters: durable memory is keyed by the subject, so a fixed
#: one would mean the second run's "a durable fact crossed into a new
#: conversation" passed on facts the *previous* run stored -- a check that
#: cannot fail is not evidence. Pin `--persona` and `--subject` to demonstrate
#: the opposite, memory surviving across runs.
PERSONAS = ("Dana", "Mateo", "Ingrid", "Yusuf", "Priya", "Noah",
            "Alma", "Tomas", "Rin", "Sofia", "Jonas", "Leila")
CITIES = ("Oslo", "Lisbon", "Nairobi", "Dublin", "Bergen", "Helsinki",
          "Porto", "Tallinn", "Seville", "Toronto", "Galway", "Gdansk")


# ── printing, because the job log is the product ─────────────────────────────


def step(number: int, title: str) -> None:
    print(f"\n{'=' * 72}\nSTEP {number}  {title}\n{'=' * 72}", flush=True)


def exchange(who: str, text: str) -> None:
    body = text.strip()
    print(f"\n[{who}] {body}", flush=True)


def conversation_state(agent: Any, conversation_id: str) -> dict[str, Any]:
    """What the server holds for a conversation: messages, summary, subject.

    The agent protocol serves this at `/v1/conversations/{id}/messages` for any
    agent with memory configured. It is the honest view of tiers 1 and 2 --
    `summary` is the folded part and `summarized_through` marks where the
    transcript and what the model actually reads diverge -- and unlike asking
    the agent what it remembers, it does not go through a language model. The
    SDK has no public wrapper for it yet, so this calls the route directly.
    """
    try:
        return agent._agent_get(f"conversations/{conversation_id}/messages") or {}
    except Exception as err:  # noqa: BLE001 - a demo reports, it does not crash
        print(f"  (could not read conversation state: {err})", flush=True)
        return {}


def report_state(agent: Any, conversation_id: str, label: str) -> dict[str, Any]:
    state = conversation_state(agent, conversation_id)
    messages = state.get("messages") or []
    summary = state.get("summary")
    print(
        f"\n  -- {label}: {len(messages)} messages held, "
        f"subject {state.get('subject') or '(none)'}",
        flush=True,
    )
    if summary:
        through = state.get("summarized_through")
        print(f"  -- rolling summary (covers through {through}):", flush=True)
        for line in str(summary).splitlines():
            print(f"       {line}", flush=True)
    else:
        print("  -- rolling summary: nothing folded yet", flush=True)
    return state


# ── the run ──────────────────────────────────────────────────────────────────


def deploy(agents: Any, args: argparse.Namespace) -> Any:
    existing = agents.get_agent(args.name)
    if args.skip_deploy:
        if existing is None:
            sys.exit(f"No agent named {args.name!r}; drop --skip-deploy to create it")
        print(f"Using the agent already deployed as {args.name!r}", flush=True)
        return existing

    entry = args.entry or str(HERE / "memory_agent.py")
    requirements = args.requirements or str(HERE / "requirements.txt")
    extra: dict[str, Any] = {}
    if args.git_url:
        # the deployment pulls memory_agent.py and prompts.py from the repository,
        # so only this script has to live in the project
        extra.update(
            git_url=args.git_url,
            git_provider=args.git_provider,
            git_branch=args.git_branch,
        )
        entry = args.entry or "agent_memory/memory_agent.py"
        requirements = args.requirements or "agent_memory/requirements.txt"
    elif not Path(entry).exists():
        sys.exit(
            f"{entry} not found. Put memory_agent.py beside this script, or pass "
            "--git-url to deploy the agent from the repository."
        )

    print(f"Deploying {entry} as {args.name!r} ...", flush=True)
    agent = agents.deploy_agent(
        entry,
        name=args.name,
        requirements=requirements,
        environment=args.environment,
        description="Agent memory demo, deployed by demo_job.py",
        tracing={"enabled": True},
        **extra,
    )
    set_model_key(agent, args)
    return agent


def set_model_key(agent: Any, args: argparse.Namespace) -> None:
    """Copy the model key from a project secret onto the deployment.

    A deployment that cannot reach a model answers every message with an error,
    which makes for a confusing demo. The key stays in the project's secret
    store; this only puts it where the pod can read it.
    """
    if not args.model_secret:
        return
    try:
        import hopsworks

        value = hopsworks.get_secrets_api().get(args.model_secret)
    except Exception as err:  # noqa: BLE001
        print(
            f"  (no secret {args.model_secret!r} ({err}); the deployment must "
            "already have its model key)",
            flush=True,
        )
        return
    try:
        deployment = agent.deployment
        env = dict(deployment.predictor.env_vars or {})
        if env.get(args.model_env) == value:
            return
        env[args.model_env] = value
        deployment.predictor.env_vars = env
        deployment.save()
        print(f"  set {args.model_env} on the deployment from secret "
              f"{args.model_secret!r}", flush=True)
    except Exception as err:  # noqa: BLE001
        print(f"  (could not set {args.model_env} on the deployment: {err})", flush=True)


def ensure_running(agent: Any, timeout: int) -> None:
    if agent.is_running():
        print("Already running.", flush=True)
        return
    print(f"Starting (up to {timeout}s) ...", flush=True)
    agent.start(await_running=timeout)
    # starting returns as soon as the instance is up; the first request still
    # pays for the import of the model SDK
    for _ in range(10):
        if agent.is_running():
            break
        time.sleep(3)
    print("Running.", flush=True)


def say(agent: Any, text: str, *, subject: str, conversation_id: str | None,
        timeout: float) -> tuple[str, str]:
    exchange("user", text)
    reply = agent.chat(
        text, conversation_id=conversation_id, subject=subject, timeout=timeout
    )
    exchange("agent", reply.text or "(no text in the reply)")
    return reply.text or "", reply.conversation_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="memorydemo", help="deployment name")
    parser.add_argument("--entry", default=None, help="agent entry script")
    parser.add_argument("--requirements", default=None)
    parser.add_argument("--environment", default="python-agent-pipeline")
    parser.add_argument("--git-url", default=None,
                        help="deploy the agent from this repository instead of local files")
    parser.add_argument("--git-provider", default="GitHub")
    parser.add_argument("--git-branch", default="main")
    parser.add_argument("--project", default=None)
    parser.add_argument("--persona", default=None,
                        help="the person the run is about; a random one when omitted")
    parser.add_argument("--city", default=None,
                        help="where they live, the durable fact the run follows; random when omitted")
    parser.add_argument("--subject", default=None,
                        help="who the conversations are with; durable memory is keyed by it. "
                             "Derived from the persona and unique per run unless you pin it, "
                             "which is how you show memory surviving across runs.")
    parser.add_argument("--model-secret", default="OPENAI_API_KEY",
                        help="project secret holding the model key; '' to skip")
    parser.add_argument("--model-env", default="OPENAI_API_KEY",
                        help="environment variable the deployment reads it as")
    parser.add_argument("--skip-deploy", action="store_true",
                        help="talk to an agent that is already deployed")
    parser.add_argument("--stop", action="store_true",
                        help="stop the deployment when the demo finishes")
    parser.add_argument("--timeout", type=int, default=600,
                        help="seconds to wait for the deployment to start")
    parser.add_argument("--reply-timeout", type=float, default=180.0)
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any tier did not do its job")
    args = parser.parse_args()

    if not os.environ.get("HOPSWORKS_API_KEY") and not os.environ.get("REST_ENDPOINT"):
        sys.exit("set HOPSWORKS_HOST, HOPSWORKS_PROJECT and HOPSWORKS_API_KEY, "
                 "or run this inside a Hopsworks job")

    import hopsworks

    project = hopsworks.login(project=args.project)
    agents = project.get_agent_serving()

    step(1, "Deploy the agent")
    agent = deploy(agents, args)
    ensure_running(agent, args.timeout)
    print(f"\n  {agent.name} at {agent.url}", flush=True)

    persona = args.persona or random.choice(PERSONAS)
    city = args.city or random.choice(CITIES)
    # unique even when the same persona comes up twice
    subject = args.subject or f"{persona.lower()}.{uuid.uuid4().hex[:8]}@example.com"
    chat = lambda text, cid: say(  # noqa: E731 - one short partial, used ten times
        agent, text, subject=subject, conversation_id=cid, timeout=args.reply_timeout
    )

    step(2, "First conversation: tell it things")
    print(f"  This run is {persona}, from {city}, filed under {subject}.", flush=True)
    answer, first = chat(
        f"I'm {persona}, I live in {city} and I'm vegetarian.", None
    )
    chat("Just for now, I'm comparing the 14-inch and the 16-inch laptop.", first)
    buffer_answer, _ = chat("Remind me which two sizes I'm comparing?", first)
    report_state(agent, first, "after three turns")

    step(3, "Keep talking, until the older turns fold into a summary")
    for message in (
        "What's a good vegetarian dish to cook on a weeknight?",
        "And something that keeps for lunch the next day?",
        "Thanks. Anything Norwegian?",
    ):
        chat(message, first)
    state = report_state(agent, first, "after six turns")
    summary = str(state.get("summary") or "")

    step(4, "A new conversation, the same person")
    print("  New conversation id, no shared history: only durable memory can carry "
          "anything across.", flush=True)
    across, second = chat("What do you remember about me?", None)
    session_leak, _ = chat("What laptop sizes was I comparing?", second)
    report_state(agent, second, "the new conversation")

    step(5, "Forget something, and check it is gone")
    chat("Please forget where I live.", second)
    after_forget, _ = chat("Where do I live?", second)

    step(6, "What each tier did")
    checks = [
        ("buffer answered from this conversation",
         "14" in buffer_answer or "16" in buffer_answer, buffer_answer),
        ("older turns folded into a rolling summary", bool(summary), summary),
        ("a durable fact crossed into a new conversation",
         city.lower() in across.lower() or "vegetarian" in across.lower(), across),
        ("the session note did not cross",
         "14" not in session_leak and "16" not in session_leak, session_leak),
        ("the forgotten fact is gone",
         city.lower() not in after_forget.lower(), after_forget),
    ]
    failed = 0
    for label, ok, evidence in checks:
        print(f"  [{'ok' if ok else '--'}] {label}", flush=True)
        if not ok:
            failed += 1
            print(f"        saw: {str(evidence).strip()[:160]}", flush=True)
    print(f"\n  conversations: {first} then {second}", flush=True)
    print(f"  traces and feedback: the deployment's Traces tab, subject {subject}",
          flush=True)

    if args.stop:
        print("\nStopping the deployment ...", flush=True)
        agent.stop()

    if failed and args.strict:
        sys.exit(f"{failed} of {len(checks)} checks did not hold")
    print(f"\nDone: {len(checks) - failed} of {len(checks)} checks held.", flush=True)


if __name__ == "__main__":
    main()
