# Minimal agent

The smallest agent that can be evaluated: `agent.py` echoes what it is asked and
calls no model; `evaluation/checks/minimal_checks.py` declares three checks with
`@check`. Deploy it git-backed from this repository and name the checks file in a
suite as `git: "deployment"`, or from any project as
`git: {"url": "https://github.com/gibchikafa/hopsragbenchagent.git", "ref": "main"}`
with `path: minimal/evaluation/checks/minimal_checks.py`. The platform's workflow
tests use it for exactly that.
