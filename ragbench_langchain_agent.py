"""
RAGBench LangChain + LangGraph agent — FastAPI + uvicorn, following the Hopsworks agent pattern.
Uses langgraph.prebuilt.create_react_agent with MySQL-backed conversation memory.

Conversation history is stored in MySQL (table: ragbench_chat_history).
Each independent conversation is identified by a session_id sent in the request.

Deploy:
    hops agent create ragbench_langchain_agent.py --name ragbenchlcagent \
        --requirements ragbench_langchain_requirements.txt \
        --environment python-agent-pipeline-meb10000-v1
    hops agent start ragbenchlcagent --wait 600

Query (single-turn):
    hops agent query ragbenchlcagent --data '{"prompt": "What is chain-of-thought prompting?"}'

Query (multi-turn — same session_id continues the conversation):
    curl -s -X POST http://10.114.123.120/v1/g2/ragbenchlcagent/query \
      -H "Authorization: ApiKey appapikey" \
      -H "Content-Type: application/json" \
      -d '{"prompt": "What is it?", "session_id": "user-abc-session-1"}'

    Response:
    {"answer": "...", "sources": [...], "session_id": "user-abc-session-1"}
"""

import logging
import os
import uuid

import hopsworks
import uvicorn
from fastapi import FastAPI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column, Index, Integer, MetaData, String, Table, Text, create_engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FG_NAME = "ragbench_embeddings"
FG_VERSION = 1
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6
MAX_HISTORY_MESSAGES = 50  # last 50 messages = 25 conversation turns


def _build_tracer_provider():
    # The platform injects this env var (and runs the OTLP sidecar) only when
    # tracing is enabled on the deployment. Without it there is nothing
    # listening on localhost:4318 — exporting would just spam connection
    # errors on every request.
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if endpoint is None:
        log.info("Tracing disabled (no OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)")
        return None
    tp = trace_sdk.TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    return tp


# ── MySQL chat store ──────────────────────────────────────────────────────────

class MySQLChatStore:
    def __init__(self, url: str, table_name: str, max_messages: int = MAX_HISTORY_MESSAGES):
        self._engine = create_engine(url, pool_pre_ping=True)
        self._max = max_messages
        # In-memory cache: session_id → list[BaseMessage] (avoids a DB round-trip
        # on every message for the lifetime of this pod).
        self._cache: dict[str, list[BaseMessage]] = {}
        metadata = MetaData()
        self._table = Table(
            table_name, metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String(255), nullable=False),
            Column("role", String(16), nullable=False),
            Column("content", Text, nullable=False),
            Index(f"idx_{table_name}_session", "session_id"),
        )
        metadata.create_all(self._engine)
        log.info("Chat history table: %s", table_name)

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        if session_id in self._cache:
            return self._cache[session_id]
        with self._engine.connect() as conn:
            rows = conn.execute(
                self._table.select()
                .where(self._table.c.session_id == session_id)
                .order_by(self._table.c.id.desc())
                .limit(self._max)
            ).fetchall()
        messages: list[BaseMessage] = [
            HumanMessage(content=r.content) if r.role == "user" else AIMessage(content=r.content)
            for r in reversed(rows)  # restore chronological order
        ]
        self._cache[session_id] = messages
        return messages

    def add_message(self, session_id: str, role: str, content: str):
        with self._engine.begin() as conn:
            conn.execute(self._table.insert().values(
                session_id=session_id, role=role, content=content
            ))
        msg: BaseMessage = HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        session_cache = self._cache.setdefault(session_id, [])
        session_cache.append(msg)
        if len(session_cache) > self._max:
            self._cache[session_id] = session_cache[-self._max:]


# ── Predictor ─────────────────────────────────────────────────────────────────

class RagbenchLCPredictor:
    def __init__(self):
        self._tracer_provider = _build_tracer_provider()
        if self._tracer_provider is not None:
            LangChainInstrumentor().instrument(tracer_provider=self._tracer_provider)

        # ── feature store ────────────────────────────────────────────────────
        project = hopsworks.login()
        self._fs = project.get_feature_store()

        # ── MySQL chat store (env vars + Hopsworks secret) ───────────────────
        secret_name = os.environ["MYSQL_PASSWORD_SECRET_NAME"]
        password = hopsworks.get_secrets_api().get(secret_name)
        user = os.environ["MYSQL_USER"]
        host = os.environ["MYSQL_HOST"]
        port = os.environ.get("MYSQL_PORT", "3306")
        db = os.environ["MYSQL_DB"]
        mysql_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
        deployment_id = os.environ.get("DEPLOYMENT_ID", "local")
        table_name = f"ragbench_chat_history_{deployment_id}"
        self._chat_store = MySQLChatStore(mysql_url, table_name=table_name)
        log.info("MySQL chat store ready: %s@%s:%s/%s", user, host, port, db)

        self._fg = None
        self._col_names = None
        log.info("Hopsworks connected. Feature group will be loaded on first query.")

        # ── embedding model ──────────────────────────────────────────────────
        self._embed = SentenceTransformer(EMBEDDING_MODEL)

        # ── sources accumulated across all tool calls within one query ───────
        self._current_sources: list[dict] = []

        # ── RAG tool ─────────────────────────────────────────────────────────
        @tool
        def search_papers(query: str) -> str:
            """Search the RAGBench academic paper corpus for passages relevant to the query.
            Returns the top matching excerpts with paper titles and similarity scores.
            Use this whenever you need factual information about AI/ML research topics."""
            if self._fg is None:
                self._fg = self._fs.get_feature_group(FG_NAME, version=FG_VERSION)
                if self._fg is None:
                    return "Feature group not available yet — run the feature pipeline first."
                self._col_names = [f.name for f in self._fg.features]
                log.info("Feature group loaded. Columns: %s", self._col_names)
            vec = self._embed.encode(query, normalize_embeddings=True).tolist()
            results = self._fg.find_neighbors(vec, col="embedding", k=TOP_K)
            if not results:
                return "No relevant passages found."
            parts = []
            for score, values in results:
                row = dict(zip(self._col_names, values))
                title = row.get("title", "").strip()
                doc_id = row.get("doc_id", "").strip()
                text = row.get("section_text", "").strip()
                parts.append(f"[{title}] (score={score:.3f})\n{text}")
                existing = next((s for s in self._current_sources if s["doc_id"] == doc_id), None)
                if existing is None:
                    self._current_sources.append({"title": title, "doc_id": doc_id, "score": round(score, 4)})
                elif score > existing["score"]:
                    existing["score"] = round(score, 4)
            return "\n\n---\n\n".join(parts)

        # ── LLM ─────────────────────────────────────────────────────────────
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            temperature=0.0,
        )

        # ── ReAct agent (LangGraph, stateless — memory injected per request) ─
        self.agent = create_react_agent(llm, [search_papers])

    def predict(self, inputs: dict) -> dict:
        prompt = inputs.get("prompt", inputs.get("question", "")).strip()
        session_id = inputs.get("session_id") or str(uuid.uuid4())
        if not prompt:
            return {"error": "No 'prompt' field in request."}

        past_messages = self._chat_store.get_messages(session_id)
        log.info("SESSION %s: %d prior messages, QUERY: %s", session_id, len(past_messages), prompt)

        self._current_sources = []
        all_messages = past_messages + [HumanMessage(content=prompt)]
        result = self.agent.invoke({"messages": all_messages})
        answer = result["messages"][-1].content

        self._chat_store.add_message(session_id, "user", prompt)
        self._chat_store.add_message(session_id, "assistant", answer)

        sources = sorted(self._current_sources, key=lambda s: s["score"], reverse=True)
        log.info("ANSWER: %s  SOURCES: %d", answer[:200], len(sources))
        return {"answer": answer, "sources": sources, "session_id": session_id}


predictor = RagbenchLCPredictor()

agent_app = FastAPI()


@agent_app.post("/query")
def query(payload: dict):
    return predictor.predict(payload)


if __name__ == "__main__":
    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
