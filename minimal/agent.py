"""The smallest agent that can be evaluated: it echoes, calls no model, and declares checks.

Deployed git-backed from this repository, it is what the platform's workflow
tests use to exercise code checks read from a repository and from the
deployment's own repository. It is also the shortest example of the shape:
an `AgentApp`, one handler, and `evaluation/checks/` beside it.
"""

from hopsworks_agents.protocol import AgentApp, AgentResponse


STEP_NAME = "minimal_step"

app = AgentApp(
    name="Minimal agent",
    description="Echoes what it is asked, calling no model.",
    tool_events=True,
)


@app.chat
async def chat(request, ctx):
    await ctx.emit_event(STEP_NAME, status="running", event_id="minimal-step-1")
    await ctx.emit_event(STEP_NAME, status="done", event_id="minimal-step-1")
    return AgentResponse.text(
        text=f"echo: {request.text or ''}",
        usage={"input_tokens": 3, "output_tokens": 2},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
