"""
FastAPI deployment server — exposes the full pipeline as a REST API.
Run: uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
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
import uvicorn

from src.pipeline.chat_engine import get_engine

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindBot — Mental Health Chatbot API",
    description="RAG-based mental health support chatbot with language detection, emotion classification, intent routing, and retrieval-augmented generation.",
    version="1.0.0",
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
    message: str = Field(..., min_length=1, max_length=2000, example="I've been feeling very anxious lately.")
    session_id: str = Field(default="default", example="user_42")


class SourceDoc(BaseModel):
    text: str
    score: float


class ChatResponse(BaseModel):
    response: str
    language: str
    language_name: str
    intent: str
    emotion: str | None
    crisis: bool
    sources: list[SourceDoc]
    elapsed_ms: float
    model: str = "pipeline"


class HealthResponse(BaseModel):
    status: str
    modules: dict


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check which modules are loaded and ready."""
    if _engine is None:
        return HealthResponse(status="loading", modules={})
    return HealthResponse(status="ok", modules=_engine.status.summary())


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Process a user message through the full NLP pipeline.

    - **Module 1**: Detects language (TF-IDF + LinearSVC)
    - **Module 2**: Classifies emotion (DistilBERT)
    - **Module 3**: Classifies intent (few-shot LLM)
    - **Module 4**: Retrieves context and generates response (Qdrant + Groq)
    """
    if _engine is None:
        raise HTTPException(503, "Engine is still loading. Retry in a moment.")

    result = _engine.process(req.message, session_id=req.session_id)
    return ChatResponse(
        response=result["response"],
        language=result["language"],
        language_name=result["language_name"],
        intent=result["intent"],
        emotion=result.get("emotion"),
        crisis=result["crisis"],
        sources=[SourceDoc(**s) for s in result.get("sources", [])],
        elapsed_ms=result["elapsed_ms"],
    )


@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """Clear conversation history for a given session."""
    if _engine:
        _engine.clear_history(session_id)
    return {"cleared": session_id}


@app.get("/", tags=["System"])
async def root():
    return {"message": "MindBot API is running. Visit /docs for interactive documentation."}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
