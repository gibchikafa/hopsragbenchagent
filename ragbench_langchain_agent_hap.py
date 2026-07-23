"""
RAGBench LangChain agent — Hopsworks Agent Protocol edition.

Same agent as ragbench_langchain_agent.py (LangGraph ReAct agent, MySQL-backed
conversation memory, RAG over the ragbench_embeddings feature group), but
served through the Hopsworks Agent Protocol so the Hopsworks UI chat panel
detects and talks to it with zero configuration:

    GET  /.well-known/hopsworks-agent.json   manifest (auto-detection + UI hints)
    POST /v1/chat                            ChatRequest -> ChatResponse
    POST /v1/chat/stream                     SSE (degrades to one completed event)
    GET  /health                             liveness

The heavy lifting is reused verbatim: importing ragbench_langchain_agent
builds the predictor (Hopsworks login, MySQL chat store, embedding model,
LangGraph agent) exactly once. The protocol's conversation_id doubles as the
chat store session_id, and retrieved papers are returned as citations.

Deploy:
    hops agent create ragbench_langchain_agent_hap.py --name ragbenchhapagent \
        --requirements ragbench_langchain_hap_requirements.txt \
        --environment python-agent-pipeline-meb10000-v1
    hops agent start ragbenchhapagent --wait 600
"""

import asyncio

from hopsworks_agent_protocol import AgentApp, AgentError, AgentResponse

# Reuses the existing predictor: constructing it connects to Hopsworks, the
# MySQL chat store, and loads the embedding model (module-level, once).
from ragbench_langchain_agent import predictor

agent_app = AgentApp(
    name="RAGBench agent",
    description="RAG agent over the RAGBench academic paper corpus "
    "(LangGraph ReAct + Hopsworks feature store retrieval).",
    welcome_message="Ask me about AI/ML research — I search the RAGBench paper corpus.",
    suggested_prompts=[
        "What is chain-of-thought prompting?",
        "What is the transformer architecture?",
        "How does retrieval-augmented generation work?",
    ],
    placeholder="Ask about AI/ML research...",
)


@agent_app.chat
async def chat(request):
    if not request.text:
        raise AgentError(
            "The message content cannot be empty.",
            code="invalid_request",
            status_code=400,
        )

    # predict() is blocking (LLM + retrieval); keep the event loop free
    result = await asyncio.to_thread(
        predictor.predict,
        {"prompt": request.text, "session_id": request.conversation_id},
    )

    if "error" in result:
        raise AgentError(result["error"], code="agent_error", status_code=400)

    return AgentResponse.text(
        text=result["answer"],
        conversation_id=result.get("session_id", request.conversation_id),
        citations=result.get("sources", []),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
