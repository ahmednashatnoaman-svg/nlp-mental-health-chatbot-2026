"""
FastAPI deployment server — exposes the full pipeline as a REST API.
Run: uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

v2 changes:
  - SourceDoc expanded with context, response, chunk_idx fields
  - ChatResponse includes crisis_level, lang_conf, language_name
  - /session/{id} GET endpoint for history inspection
  - Source serialization uses explicit field extraction (no **s unpacking)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from src.pipeline.chat_engine import get_engine

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindBot — Mental Health Chatbot API",
    description=(
        "RAG-based mental health support chatbot with language detection, "
        "emotion classification, intent routing, and retrieval-augmented generation. "
        "Supports 20 languages via automatic translation."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load engine on startup ────────────────────────────────────────────────
_engine = None

@app.on_event("startup")
async def startup():
    global _engine
    _engine = get_engine()


# ─────────────────────────────────────────────────────────────────────────────
#  REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:      str = Field(..., min_length=1, max_length=2000,
                              example="I've been feeling very anxious lately.")
    session_id:   str = Field(default="default", example="user_42")
    top_k:        int = Field(default=5, ge=1, le=10,
                              description="Number of RAG passages to retrieve")
    strong_model: bool = Field(default=True,
                               description="Use gpt-oss-120b (True) or gpt-oss-20b (False)")


class SourceDoc(BaseModel):
    text:      str
    score:     float
    context:   str = ""   # original patient question (may be empty for chunk-only records)
    response:  str = ""   # counselor answer (may be empty for chunk-only records)


class EmotionDetail(BaseModel):
    emotion:    Optional[str]  = None
    emoji:      str            = ""
    color:      str            = ""
    tone:       str            = ""
    confidence: float          = 0.0
    all_scores: dict           = {}
    device:     str            = ""


class IntentDetail(BaseModel):
    intent:     str   = ""
    emoji:      str   = ""
    color:      str   = ""
    route:      str   = ""
    confidence: float = 0.0


class ChatResponse(BaseModel):
    response:      str
    language:      str
    language_name: str
    lang_conf:     float
    intent:        str
    intent_detail: IntentDetail
    emotion:       Optional[str]
    emotion_detail: EmotionDetail
    crisis:        bool
    crisis_level:  str   # "high" | "medium" | "low"
    sources:       list[SourceDoc]
    elapsed_ms:    float
    model:         str = "pipeline/v2"


class HealthResponse(BaseModel):
    status:  str
    modules: dict


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns:      int
    history:    list[dict]


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "message": "MindBot API v2 is running.",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check which NLP modules are loaded and ready."""
    if _engine is None:
        return HealthResponse(status="loading", modules={})
    return HealthResponse(status="ok", modules=_engine.status.summary())


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Process a user message through the full NLP pipeline.

    Pipeline order:
    1. **Module 1**: Detects language (TF-IDF + LinearSVC, 20 languages)
    2. **Translation**: Translates to English for NLP processing (Groq gpt-oss-20b)
    3. **Crisis Check**: 3-level risk scoring (HIGH / MEDIUM / LOW)
    4. **Module 3**: Classifies intent (few-shot Groq)
    5. **Module 2**: Classifies emotion (DistilBERT) — mental health path only
    6. **Module 4**: Retrieves from Qdrant and generates response (gpt-oss-120b)
    7. Response is generated directly in the user's detected language
    """
    if _engine is None:
        raise HTTPException(503, "Engine is still loading. Retry in a moment.")

    result = _engine.process(
        req.message,
        session_id=req.session_id,
        top_k=req.top_k,
        strong_model=req.strong_model,
    )

    # Explicit field extraction prevents Pydantic validation errors from extra keys
    sources = [
        SourceDoc(
            text=s.get("text", ""),
            score=s.get("score", 0.0),
            context=s.get("context", ""),
            response=s.get("response", ""),
        )
        for s in result.get("sources", [])
    ]

    intent_meta   = result.get("intent_meta", {})
    emotion_meta  = result.get("emotion_meta", {})

    return ChatResponse(
        response=result["response"],
        language=result["language"],
        language_name=result["language_name"],
        lang_conf=result.get("lang_conf", 0.0),
        intent=result["intent"],
        intent_detail=IntentDetail(
            intent=intent_meta.get("intent", result["intent"]),
            emoji=intent_meta.get("emoji", ""),
            color=intent_meta.get("color", ""),
            route=intent_meta.get("route", ""),
            confidence=intent_meta.get("confidence", 0.0),
        ),
        emotion=result.get("emotion"),
        emotion_detail=EmotionDetail(
            emotion=emotion_meta.get("emotion"),
            emoji=emotion_meta.get("emoji", ""),
            color=emotion_meta.get("color", ""),
            tone=emotion_meta.get("tone", ""),
            confidence=emotion_meta.get("confidence", 0.0),
            all_scores=emotion_meta.get("all_scores", {}),
            device=emotion_meta.get("device", ""),
        ),
        crisis=result["crisis"],
        crisis_level=result.get("crisis_level", "low"),
        sources=sources,
        elapsed_ms=result["elapsed_ms"],
    )


@app.get("/session/{session_id}", response_model=SessionHistoryResponse, tags=["Session"])
async def get_session(session_id: str):
    """Retrieve conversation history for a given session."""
    if _engine is None:
        raise HTTPException(503, "Engine not ready.")
    history = _engine.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        turns=len(history),
        history=[{"role": t.role, "content": t.content} for t in history],
    )


@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """Clear conversation history for a given session."""
    if _engine:
        _engine.clear_history(session_id)
    return {"cleared": session_id}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
