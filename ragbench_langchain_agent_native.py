"""
RAGBench LangChain agent — native Hopsworks Agent Protocol implementation.

Fully standalone (no import from ragbench_langchain_agent.py): the SDK owns
the HTTP surface (manifest, /v1/chat, /v1/chat/stream, /health, CORS),
tracing (LangChain instrumentation activates automatically when tracing is
enabled on the deployment), and conversation memory (SqlChatMemory on the
project MySQL, keyed by the protocol's conversation_id). The agent code is
only the domain: retrieval tool + LangGraph ReAct agent + one streaming
handler that yields tokens as they are generated.

Deploy:
    hops agent create ragbench_langchain_agent_native.py --name ragbenchnative \
        --requirements ragbench_langchain_hap_requirements.txt \
        --environment python-agent-pipeline-meb10000-v1
    hops agent start ragbenchnative --wait 600
"""

import logging
import os

import hopsworks
from hopsworks_agent_protocol import AgentApp, AgentError, AgentResponse, SqlChatMemory  # noqa: E501
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FG_NAME = "ragbench_embeddings"
FG_VERSION = 1
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6


# ── domain setup (module level, once) ────────────────────────────────────────

project = hopsworks.login()
fs = project.get_feature_store()
embed = SentenceTransformer(EMBEDDING_MODEL)

_fg = None
_col_names: list[str] | None = None
# sources accumulated across tool calls within one request (single worker,
# one request at a time — same simplification as the original agent)
_current_sources: list[dict] = []


@tool
def search_papers(query: str) -> str:
    """Search the RAGBench academic paper corpus for passages relevant to the query.
    Returns the top matching excerpts with paper titles and similarity scores.
    Use this whenever you need factual information about AI/ML research topics."""
    global _fg, _col_names
    if _fg is None:
        _fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
        if _fg is None:
            return "Feature group not available yet — run the feature pipeline first."
        _col_names = [f.name for f in _fg.features]
        log.info("Feature group loaded. Columns: %s", _col_names)
    vec = embed.encode(query, normalize_embeddings=True).tolist()
    results = _fg.find_neighbors(vec, col="embedding", k=TOP_K)
    if not results:
        return "No relevant passages found."
    parts = []
    for score, values in results:
        row = dict(zip(_col_names, values))
        title = row.get("title", "").strip()
        doc_id = row.get("doc_id", "").strip()
        text = row.get("section_text", "").strip()
        parts.append(f"[{title}] (score={score:.3f})\n{text}")
        existing = next((s for s in _current_sources if s["doc_id"] == doc_id), None)
        if existing is None:
            _current_sources.append(
                {"title": title, "doc_id": doc_id, "score": round(score, 4)}
            )
        elif score > existing["score"]:
            existing["score"] = round(score, 4)
    return "\n\n---\n\n".join(parts)


llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024, temperature=0.0)
agent = create_react_agent(llm, [search_papers])


# ── the protocol app: manifest + endpoints + tracing + memory ────────────────

agent_app = AgentApp(
    name="RAGBench agent (native)",
    description="RAG agent over the RAGBench academic paper corpus "
    "(LangGraph ReAct + Hopsworks feature store retrieval).",
    framework="langgraph",
    welcome_message="Ask me about AI/ML research — I search the RAGBench paper corpus.",
    suggested_prompts=[
        "What is chain-of-thought prompting?",
        "What is the transformer architecture?",
        "How does retrieval-augmented generation work?",
    ],
    placeholder="Ask about AI/ML research...",
    # zero-config: project MySQL from the platform-injected MYSQL_* env
    # vars, table name derived from DEPLOYMENT_ID
    memory=SqlChatMemory(),
)


@agent_app.stream
async def stream(request):
    """One handler serves both endpoints: /v1/chat/stream emits each yielded
    token as a message.delta; /v1/chat collects them into a single reply."""
    if not request.text:
        raise AgentError(
            "The message content cannot be empty.",
            code="invalid_request",
            status_code=400,
        )

    # history: recorded automatically by the SDK after each turn, in the
    # {"role", "content"} shape LangGraph accepts directly
    history = agent_app.memory.get(request.conversation_id)
    _current_sources.clear()

    async for event in agent.astream_events(
        {"messages": history + [HumanMessage(content=request.text)]},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            # Anthropic content is a string or a list of content blocks
            if isinstance(chunk, str):
                if chunk:
                    yield chunk
            elif isinstance(chunk, list):
                for block in chunk:
                    if isinstance(block, dict) and block.get("type") == "text":
                        yield block.get("text", "")

    # citations ride on the final message.completed event
    yield AgentResponse.parts(
        conversation_id=request.conversation_id,
        citations=sorted(_current_sources, key=lambda s: s["score"], reverse=True),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
