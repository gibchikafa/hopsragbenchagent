"""
RAGBench LlamaIndex agent — native Hopsworks Agent Protocol implementation.

Standalone (no import from llamaindex_agent.py): the SDK owns the HTTP
surface (manifest, /v1/chat, /v1/chat/stream, /health, CORS), tracing
(LlamaIndex instrumentation activates automatically when tracing is enabled on
the deployment), and memory. The agent code is only the domain: retrieval tool +
LlamaIndex ReActAgent + one streaming handler.

ctx.stream_llamaindex pipes the agent's run through, yielding text deltas and
turning tool calls into tool_event chips in the chat panel.

Memory has three tiers, all served by one store:

1. **Conversation buffer** — the turns of this conversation, keyed by the
   protocol's conversation_id. Read with ``ctx.history``.
2. **Rolling summary** — once a conversation outgrows the buffer, older turns
   are folded into a summary instead of dropping out of view. This changes what
   ``ctx.history`` means: it is the turns *since* the last fold, and everything
   older is in ``ctx.summary``. Pass both to the model — ``ctx.system_context()``
   assembles them.
3. **Durable per-user memory** — facts the agent chooses to keep across
   conversations, via the ``remember`` / ``recall`` / ``forget`` / ``search``
   tools registered below.

Deploy:
    hops agent create llamaindex_agent_native.py --name ragbenchlinative \
        --requirements llamaindex_hap_requirements.txt \
        --environment python-agent-pipeline-meb10000-v1
    hops agent start ragbenchlinative --wait 600
"""

import logging

import hopsworks
from hopsworks_agent_protocol import (  # noqa: E501
    AgentApp,
    AgentError,
    AgentResponse,
    ManagedMemoryService,
    anthropic_summarizer,
    memory_tools,
)
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

SYSTEM_PROMPT = """\
You answer questions about AI/ML research by searching the RAGBench paper \
corpus. Ground every factual claim in a passage you retrieved; say so when the \
corpus does not cover something rather than answering from memory.

You also keep notes about the person you are talking to. When they tell you \
something that will still be true next time — the topics they work on, the \
depth of explanation they want, a paper they are writing — store it with \
`remember`. Do not store the content of papers there; that is what the corpus \
is for.\
"""


# ── domain setup (module level, once) ────────────────────────────────────────

project = hopsworks.login()
fs = project.get_feature_store()
embed = SentenceTransformer(EMBEDDING_MODEL)
llm = Anthropic(model="claude-haiku-4-5", max_tokens=1024, temperature=0.0)

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


# The memory tools arrive as LlamaIndex FunctionTools; they resolve the store
# and the current user from the request context, so their signatures carry no
# plumbing. Registration is explicit — the SDK cannot reach into an arbitrary
# framework's tool list, and appending to yours behind your back would be worse
# than asking.
#
# Note the two searches are different things and the model is told so by their
# docstrings: `search_papers` searches the corpus, `search` searches what *this
# user* said in earlier conversations.
tools = [FunctionTool.from_defaults(search_papers), *memory_tools("llamaindex")]


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
    # Zero-config connection: project MySQL from the platform-injected MYSQL_*
    # env vars, tables derived from DEPLOYMENT_ID.
    memory=ManagedMemoryService(
        # Tier 2. Without a summarizer the buffer is a fixed newest-N window and
        # older turns simply stop being visible; with one they are compacted
        # into ctx.summary instead. Runs after the response has streamed, so it
        # costs request duration every Nth turn and never time-to-answer.
        summarize=anthropic_summarizer(),
        # Tier 3. Creates the scoped-state table and enables remember/recall.
        long_term=True,
        # Semantic search over past conversations is off here: it needs an
        # embedder plus a vector store, and the Hopsworks feature-group backend
        # has not been verified against a live feature store yet. The `search`
        # tool still works — it falls back to keyword matching over SQL — so
        # switching this on later needs no prompt change:
        #
        #   from hopsworks_agent_protocol import vector_store_for
        #   embedder = lambda text: embed.encode(
        #       text, normalize_embeddings=True
        #   ).tolist()
        #   ... embedder=embedder, vector_store=vector_store_for(embedder)
        #
        # (the same SentenceTransformer loaded above is reusable for it)
    ),
    tool_events=True,
    # ReActAgent is a LlamaIndex Workflow — the SDK derives its graph from the
    # @step methods (prebuilt agents show the framework's ReAct workflow;
    # custom workflows show your own step graph)
    graph=ReActAgent(tools=tools, llm=llm),
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

    # The workflow ReActAgent does not reliably consume injected memory, so the
    # conversation goes into the prompt. ctx.system_context() carries the
    # rolling summary of compacted turns plus what the agent has stored about
    # this user, and returns "" when there is nothing yet; ctx.history is the
    # turns since the last fold. Together they are the whole conversation.
    sections = [SYSTEM_PROMPT + ctx.system_context()]
    if ctx.history:
        recent = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in ctx.history
        )
        sections.append(f"Recent turns:\n{recent}")
    sections.append(f"Current message: {request.text}")
    prompt = "\n\n".join(sections)

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
