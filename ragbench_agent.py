"""
RAGBench LlamaIndex agent — FastAPI + uvicorn, following the Hopsworks agent pattern.
Conversation history is persisted in MySQL (table: ragbench_li_chat_history).
Each independent conversation is identified by a session_id sent in the request.

Deploy:
    hops agent create ragbench_agent.py --name ragbenchagent --environment ragbench-agent
    hops agent start ragbenchagent --wait 600

Query (single-turn):
    hops agent query ragbenchagent --data '{"prompt": "What is chain-of-thought prompting?"}'

Query (multi-turn — same session_id continues the conversation):
    curl -s -X POST http://10.114.123.120/v1/g2/ragbenchagent/query \
      -H "Authorization: ApiKey appapikey" \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Can you give an example?", "session_id": "<id from previous response>"}'

    Response:
    {"answer": "...", "sources": [...], "session_id": "..."}
"""

import logging
import os
import uuid

import hopsworks
import uvicorn
from fastapi import FastAPI
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column, Index, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FG_NAME = "ragbench_embeddings"
FG_VERSION = 1
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6
MAX_HISTORY_MESSAGES = 50  # last 50 messages = 25 conversation turns


def build_tracer_provider():
    endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "http://localhost:4318/v1/traces",
    )
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    return tracer_provider


# ── MySQL chat store ──────────────────────────────────────────────────────────

class MySQLChatStore:
    def __init__(self, url: str, table_name: str, max_messages: int = MAX_HISTORY_MESSAGES):
        self._engine = create_engine(url, pool_pre_ping=True)
        self._max = max_messages
        # In-memory cache: session_id → list[ChatMessage] (avoids a DB round-trip
        # on every message for the lifetime of this pod).
        self._cache: dict[str, list[ChatMessage]] = {}
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

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        if session_id in self._cache:
            return self._cache[session_id]
        with self._engine.connect() as conn:
            rows = conn.execute(
                self._table.select()
                .where(self._table.c.session_id == session_id)
                .order_by(self._table.c.id.desc())
                .limit(self._max)
            ).fetchall()
        messages = [
            ChatMessage(
                role=MessageRole.USER if r.role == "user" else MessageRole.ASSISTANT,
                content=r.content,
            )
            for r in reversed(rows)  # restore chronological order
        ]
        self._cache[session_id] = messages
        return messages

    def add_message(self, session_id: str, role: str, content: str):
        with self._engine.begin() as conn:
            conn.execute(self._table.insert().values(
                session_id=session_id, role=role, content=content
            ))
        msg = ChatMessage(
            role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
            content=content,
        )
        session_cache = self._cache.setdefault(session_id, [])
        session_cache.append(msg)
        if len(session_cache) > self._max:
            self._cache[session_id] = session_cache[-self._max:]


# ── Predictor ─────────────────────────────────────────────────────────────────

class RagbenchPredictor:
    def __init__(self):
        self.tracer_provider = build_tracer_provider()
        LlamaIndexInstrumentor().instrument(tracer_provider=self.tracer_provider)

        # ── feature store ────────────────────────────────────────────────────
        project = hopsworks.login()
        self._fs = project.get_feature_store()
        self._fg = None
        self._col_names = None
        log.info("Hopsworks connected. Feature group will be loaded on first query.")

        # ── MySQL chat store (env vars + Hopsworks secret) ───────────────────
        secret_name = os.environ["MYSQL_PASSWORD_SECRET_NAME"]
        password = hopsworks.get_secrets_api().get(secret_name)
        user = os.environ["MYSQL_USER"]
        host = os.environ["MYSQL_HOST"]
        port = os.environ.get("MYSQL_PORT", "3306")
        db = os.environ["MYSQL_DB"]
        mysql_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
        deployment_id = os.environ.get("DEPLOYMENT_ID", "local")
        table_name = f"ragbench_li_chat_history_{deployment_id}"
        self._chat_store = MySQLChatStore(mysql_url, table_name=table_name)
        log.info("MySQL chat store ready: %s@%s:%s/%s", user, host, port, db)

        # ── embedding model ──────────────────────────────────────────────────
        self._embed = SentenceTransformer(EMBEDDING_MODEL)

        # ── LLM ─────────────────────────────────────────────────────────────
        self._llm = Anthropic(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            temperature=0.0,
        )

        # ── RAG tool ─────────────────────────────────────────────────────────
        self._current_sources: list[dict] = []

        def search_papers(query: str) -> str:
            """Search the RAGBench academic paper corpus for passages relevant to the query.
            Returns the top matching excerpts with paper titles and similarity scores."""
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

        self._tools = [FunctionTool.from_defaults(search_papers)]

    async def _predict_async(self, inputs: dict) -> dict:
        prompt = inputs.get("prompt", inputs.get("question", "")).strip()
        session_id = inputs.get("session_id") or str(uuid.uuid4())
        if not prompt:
            return {"error": "No 'prompt' field in request."}

        # Load prior turns from MySQL and inject into a per-request memory buffer.
        past_messages = self._chat_store.get_messages(session_id)
        memory = ChatMemoryBuffer.from_defaults(
            chat_history=past_messages,
            token_limit=4096,
        )
        log.info("SESSION %s: %d prior messages, QUERY: %s", session_id, len(past_messages), prompt)

        self._current_sources = []
        agent = ReActAgent(tools=self._tools, llm=self._llm, memory=memory)
        result = await agent.run(prompt)
        answer = str(result)

        # Persist the new turn.
        self._chat_store.add_message(session_id, "user", prompt)
        self._chat_store.add_message(session_id, "assistant", answer)

        sources = sorted(self._current_sources, key=lambda s: s["score"], reverse=True)
        log.info("ANSWER: %s  SOURCES: %d", answer[:200], len(sources))
        return {"answer": answer, "sources": sources, "session_id": session_id}


predictor = RagbenchPredictor()

agent_app = FastAPI()

FastAPIInstrumentor.instrument_app(
    agent_app,
    tracer_provider=predictor.tracer_provider,
    excluded_urls=r"^(?!.*\/query$).*",
)


@agent_app.post("/query")
async def query(payload: dict):
    return await predictor._predict_async(payload)


if __name__ == "__main__":
    uvicorn.run(agent_app, host="0.0.0.0", port=8080)
