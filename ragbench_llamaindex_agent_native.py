"""
RAGBench LlamaIndex agent — native Hopsworks Agent Protocol implementation.

Standalone (no import from ragbench_llamaindex_agent.py): the SDK owns the HTTP
surface (manifest, /v1/chat, /v1/chat/stream, /health, CORS), tracing
(LlamaIndex instrumentation activates automatically when tracing is enabled on
the deployment), and conversation memory (SqlChatMemory on the project MySQL,
keyed by the protocol's conversation_id). The agent code is only the domain:
retrieval tool + LlamaIndex ReActAgent + one streaming handler.

ctx.stream_llamaindex pipes the agent's run through, yielding text deltas and
turning tool calls into tool_event chips in the chat panel.

Deploy:
    hops agent create ragbench_llamaindex_agent_native.py --name ragbenchlinative \
        --requirements ragbench_llamaindex_hap_requirements.txt \
        --environment python-agent-pipeline-meb10000-v1
    hops agent start ragbenchlinative --wait 600
"""

import logging

import hopsworks
from hopsworks_agent_protocol import AgentApp, AgentError, AgentResponse, SqlChatMemory  # noqa: E501
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
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
llm = Anthropic(model="claude-haiku-4-5-20251001", max_tokens=1024, temperature=0.0)

_fg = None
_col_names: list[str] | None = None
# sources accumulated across tool calls within one request (single worker,
# one request at a time — same simplification as the original agent)
_current_sources: list[dict] = []


def search_papers(query: str) -> str:
    """Search the RAGBench academic paper corpus for passages relevant to the query.
    Returns the top matching excerpts with paper titles and similarity scores."""
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


tools = [FunctionTool.from_defaults(search_papers)]


# ── the protocol app: manifest + endpoints + tracing + memory ────────────────

agent_app = AgentApp(
    name="RAGBench agent LlamaIndex (native)",
    description="RAG agent over the RAGBench academic paper corpus "
    "(LlamaIndex ReAct + Hopsworks feature store retrieval).",
    framework="llamaindex",
    welcome_message="Ask me about AI/ML research — I search the RAGBench paper corpus.",
    suggested_prompts=[
        "What is chain-of-thought prompting?",
        "What is the transformer architecture?",
        "How does retrieval-augmented generation work?",
    ],
    placeholder="Ask about AI/ML research...",
    memory=SqlChatMemory(),
    tool_events=True,
)


@agent_app.stream
async def stream(request, ctx):
    """One handler serves both endpoints. ctx.stream_llamaindex yields the
    agent's text deltas and surfaces its tool calls as chips."""
    if not request.text:
        raise AgentError(
            "The message content cannot be empty.",
            code="invalid_request",
            status_code=400,
        )

    # The workflow ReActAgent does not reliably consume injected memory, so
    # prepend the SDK-managed history into the prompt (proven approach).
    history = ctx.history
    if history:
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history]
        prompt = (
            "Conversation history:\n"
            + "\n".join(lines)
            + f"\n\nCurrent message: {request.text}"
        )
    else:
        prompt = request.text

    _current_sources.clear()
    agent = ReActAgent(tools=tools, llm=llm)

    async for delta in ctx.stream_llamaindex(agent.run(prompt)):
        yield delta

    yield AgentResponse.parts(
        conversation_id=request.conversation_id,
        citations=sorted(_current_sources, key=lambda s: s["score"], reverse=True),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
