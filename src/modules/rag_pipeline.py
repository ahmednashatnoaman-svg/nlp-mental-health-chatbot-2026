"""
Module 4 — Q&A RAG Pipeline
Qdrant Cloud (qdrant-client >= 1.7) + BAAI/bge-base-en-v1.5 (768-dim) + Groq.

Payload structure stored in Qdrant:
  context       – patient question text
  response      – counselor answer text
  chunk         – embedded text segment (may be partial)
  chunk_idx     – chunk sequence within source document
  total_chunks  – total chunks for the source document
  source_row_id – original dataset row number
"""
import os
import torch
from pathlib import Path

from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "health-counseling-dataset"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"   # 768-dim — matches stored vectors
VECTOR_DIM       = 768
STRONG_MODEL     = "openai/gpt-oss-120b"
WEAK_MODEL       = "openai/gpt-oss-20b"


def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a compassionate, professional mental health support assistant.
Your role: deliver thorough, empathetic, organized, and deeply helpful responses.

CORE RULES:
1. Base your answer PRIMARILY on the provided knowledge-base context.
2. Respond in the SAME LANGUAGE as the user's message (specified below).
3. Match the empathy tone specified. Never be clinical or cold.
4. NEVER diagnose. NEVER prescribe medication.
5. For serious concerns always encourage professional help.
6. NEVER say: "Just think positive", "Others have it worse", "Stop overthinking".
7. Use markdown formatting: bold headers (**Section:**), bullet points (•), numbered steps.

RESPONSE STRUCTURE — follow this template for every mental health question:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
**💙 I hear you**
2-3 sentences warmly acknowledging the specific emotion or situation shared. Validate
the feeling without judgment. Make the person feel truly seen and understood.

**🔍 Understanding what you're experiencing**
2-3 sentences explaining what they might be going through from a psychological/emotional
perspective. Use accessible, compassionate language — not clinical jargon. Draw on the
knowledge-base context to explain the experience.

**🛠️ Practical strategies to help**
4-6 concrete, actionable strategies presented as numbered steps or bullet points.
Each strategy should be explained with enough detail to be immediately useful.
Base these on the counseling knowledge base provided.

**🌱 Building on your strengths**
2-3 sentences acknowledging their resilience or positive qualities. Frame challenges
as opportunities for growth when appropriate.

**🤝 Professional support**
1-2 sentences encouraging professional help when the situation warrants it, without
making it sound dismissive or alarming.

**💬 I'm here for you**
1-2 sentences of warm, personal encouragement. Invite them to share more.
━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: For positive emotions (happiness, joy, excitement), still use the structure
but adapt it — celebrate with them, explore what's contributing to the positive state,
and offer ways to sustain and deepen wellbeing.

Your goal: make the user feel deeply heard, genuinely understood, and meaningfully supported."""

_RAG_TEMPLATE = """{system}

USER EMOTION: {emotion} — respond with a {tone} tone.
RESPOND IN: {language_name}

{history_block}
RELEVANT COUNSELING KNOWLEDGE BASE:
{context}

CURRENT USER MESSAGE: {question}

Use the conversation history above (if any) to personalize your response.
Reference earlier topics when relevant — show the user you remember them.
Follow the structured template above. Use markdown bold headers and bullet points.
RESPOND FULLY IN {language_name}. Write as much as needed — do not cut short.

RESPONSE:"""

_REWRITE_TEMPLATE = """Rewrite this message as a concise mental health knowledge-base search query.
Remove first-person pronouns, focus on the clinical/emotional topic. ONE short line only.

Message: {message}
Search query:"""

_DIRECT_TEMPLATE = """{system}
LANGUAGE: {language_name}
INTENT: {intent}
{history_block}
USER MESSAGE: {message}

Reply warmly in {language_name}. Be genuinely engaging, caring, and brief (3-5 sentences).
Reference the conversation history above if relevant."""


def _format_history(turns: list[dict]) -> str:
    """
    Format conversation history for injection into prompts.
    Each entry: {"role": "user"|"assistant", "content": str}
    Assistant turns are truncated to 600 chars to control context size.
    """
    if not turns:
        return ""
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"CONVERSATION HISTORY ({len(turns)} turns):",
    ]
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        content = t["content"].strip()
        # Truncate long assistant responses to keep prompt size manageable
        if role == "Assistant" and len(content) > 600:
            content = content[:600] + "…"
        lines.append(f"{role}: {content}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines) + "\n\n"


def _format_payload(payload: dict) -> str:
    """
    Build a readable text block from a Qdrant payload.
    Handles all payload variants from the existing collection.
    """
    context  = payload.get("context",  "") or ""
    response = payload.get("response", "") or ""
    chunk    = payload.get("chunk",    "") or ""

    if context and response:
        # Best case: full Q&A pair
        return f"Patient: {context.strip()}\nCounselor: {response.strip()}"
    if chunk:
        return chunk.strip()
    # Fallback: any non-empty value
    return " ".join(str(v) for v in payload.values() if v).strip()[:500]


class RAGPipeline:
    def __init__(
        self,
        groq_api_key:   str | None = None,
        qdrant_url:     str | None = None,
        qdrant_api_key: str | None = None,
        embed_model:    str        = EMBED_MODEL_NAME,
    ):
        groq_key = groq_api_key  or os.getenv("LLM_API_KEY")   or os.getenv("GROQ_API_KEY")
        q_url    = qdrant_url    or os.getenv("QDRANT_CLUSTER_ENDPOINT") or os.getenv("QDRANT_URL")
        q_key    = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not groq_key:
            raise ValueError("LLM_API_KEY not set — add it to .env")
        if not q_url:
            raise ValueError("QDRANT_CLUSTER_ENDPOINT not set — add it to .env")

        self._groq     = Groq(api_key=groq_key)
        self._qdrant   = QdrantClient(url=q_url, api_key=q_key, timeout=60.0)
        self._device   = _best_device()
        self._embedder = SentenceTransformer(embed_model, device=self._device)

    # ── Query rewriting ────────────────────────────────────────────────────────
    def rewrite_query(self, query: str) -> str:
        """LLM-rewrite user query for better retrieval signal."""
        try:
            r = self._groq.chat.completions.create(
                model=WEAK_MODEL,
                messages=[{"role": "user", "content": _REWRITE_TEMPLATE.format(message=query)}],
                temperature=0.0, max_tokens=80,
            )
            rewritten = r.choices[0].message.content.strip().split("\n")[0]
            return rewritten if len(rewritten) > 5 else query
        except Exception:
            return query

    # ── Retrieval ──────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 5, rewrite: bool = True) -> list[dict]:
        """Embed → (optionally rewrite) → Qdrant search → formatted results."""
        search_query = self.rewrite_query(query) if rewrite else query
        vec = self._embedder.encode(
            search_query, normalize_embeddings=True
        ).tolist()

        result = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=top_k,
            with_payload=True,
        )
        docs = []
        for pt in result.points:
            payload = pt.payload or {}
            text    = _format_payload(payload)
            if text:
                docs.append({
                    "text":      text,
                    "score":     round(pt.score, 4),
                    "context":   (payload.get("context")  or "")[:300],
                    "response":  (payload.get("response") or "")[:300],
                    "chunk_idx": payload.get("chunk_idx"),
                })
        return sorted(docs, key=lambda x: x["score"], reverse=True)

    # ── Generation ─────────────────────────────────────────────────────────────
    def answer(
        self,
        question:             str,
        emotion:              str        = "neutral",
        tone:                 str        = "empathetic",
        language:             str        = "en",
        language_name:        str        = "English",
        top_k:                int        = 5,
        use_strong_model:     bool       = True,
        conversation_ctx:     str        = "",
        conversation_history: list[dict] = None,  # full turn log for memory
    ) -> dict:
        """
        Full RAG: rewrite → retrieve → generate.

        conversation_history: list of {"role": "user"|"assistant", "content": str}
        representing the full conversation so far (excluding current message).
        The LLM receives this as an explicit memory block so it can reference
        earlier topics, emotions, and advice given in previous turns.
        """
        # Augment retrieval query with the most recent user turn for context
        retrieval_query = (
            f"{conversation_ctx} {question}".strip() if conversation_ctx else question
        )
        docs = self.retrieve(retrieval_query, top_k=top_k, rewrite=True)

        context = "\n\n---\n\n".join(
            f"[Relevance {d['score']:.2f}]\n{d['text']}"
            for d in docs
        ) if docs else "No relevant passages found in the knowledge base."

        history_block = _format_history(conversation_history or [])

        prompt = _RAG_TEMPLATE.format(
            system=_SYSTEM_PROMPT,
            emotion=emotion,
            tone=tone,
            language_name=language_name,
            history_block=history_block,
            context=context,
            question=question,
        )
        model = STRONG_MODEL if use_strong_model else WEAK_MODEL
        r = self._groq.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.65,
            # max_tokens intentionally omitted — let the model decide its own length
        )
        return {
            "response":        r.choices[0].message.content.strip(),
            "sources":         docs,
            "model":           model,
            "retrieved_count": len(docs),
            "rewritten_query": retrieval_query,
        }

    def direct_reply(
        self,
        message:              str,
        intent:               str        = "greeting",
        language_name:        str        = "English",
        conversation_history: list[dict] = None,  # memory for direct replies too
    ) -> dict:
        """LLM reply without RAG (greetings, farewells, gratitude, out-of-scope).
        Receives conversation history so it can greet by name, reference prior topics, etc."""
        history_block = _format_history(conversation_history or [])
        prompt = _DIRECT_TEMPLATE.format(
            system=_SYSTEM_PROMPT,
            language_name=language_name,
            intent=intent,
            history_block=history_block,
            message=message,
        )
        r = self._groq.chat.completions.create(
            model=WEAK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            # max_tokens intentionally omitted — no artificial cap
        )
        return {
            "response":        r.choices[0].message.content.strip(),
            "sources":         [],
            "model":           WEAK_MODEL,
            "retrieved_count": 0,
        }

    # ── Ingestion ──────────────────────────────────────────────────────────────
    def create_collection(self) -> None:
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    def ingest(self, chunks: list[str], batch_size: int = 64) -> int:
        import uuid
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            embs  = self._embedder.encode(
                batch, normalize_embeddings=True, show_progress_bar=True
            )
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=e.tolist(),
                    payload={"chunk": t},
                )
                for t, e in zip(batch, embs)
            ]
            self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            total += len(points)
        return total

    def collection_count(self) -> int:
        try:
            return self._qdrant.get_collection(COLLECTION_NAME).points_count or 0
        except Exception:
            return 0

    def is_ready(self) -> bool:
        return self.collection_count() > 0


def get_pipeline(**kwargs) -> RAGPipeline:
    return RAGPipeline(**kwargs)
