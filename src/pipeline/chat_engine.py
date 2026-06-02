"""
Chat Engine — Orchestrates all 4 NLP modules end-to-end.

Pipeline:
  1. Crisis detection (bonus — HIGH level overrides everything)
  2. Language detection (Module 1)
  3. Translation to English if needed (bonus)
  4. Intent classification with emotion hint (Module 3)
  5. Emotion classification (Module 2)  — only for mental health queries
  6. RAG answer with conversation context (Module 4)  OR  direct LLM reply
  7. Translate response back if needed (bonus)

Single entry-point: engine.process(message, session_id)
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.modules.language_detector  import LanguageDetector
from src.modules.emotion_classifier  import EmotionClassifier, EMOTION_META
from src.modules.intent_classifier   import IntentClassifier
from src.modules.rag_pipeline        import RAGPipeline
from src.utils.crisis_detector       import detect_crisis

# ── Session ────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role:    str          # "user" | "assistant"
    content: str
    meta:    dict = field(default_factory=dict)


_sessions: dict[str, list[Turn]] = defaultdict(list)
MAX_HISTORY = 10   # turns kept per session


# ── Module status ──────────────────────────────────────────────────────────────

class ModuleStatus:
    def __init__(self):
        self.language = False
        self.emotion  = False
        self.intent   = False
        self.rag      = False
        self.errors:  dict[str, str] = {}

    def all_ready(self) -> bool:
        return self.language and self.emotion and self.intent and self.rag

    def summary(self) -> dict:
        return {
            "language": self.language, "emotion": self.emotion,
            "intent":   self.intent,   "rag":     self.rag,
            "errors":   self.errors,
        }


# ── Translation helper ─────────────────────────────────────────────────────────

def _translate(text: str, direction: str, language: str, groq_client) -> str:
    """
    direction='to_en':   translate <language> text → English
    direction='from_en': translate English text → <language>
    Returns original text if language is 'en' or translation fails.
    """
    if language == "en" or not text.strip():
        return text
    try:
        if direction == "to_en":
            prompt = f"Translate this {language} text to English. Reply with ONLY the translation:\n\n{text}"
        else:
            from src.modules.language_detector import LANGUAGE_NAMES
            lang_name = LANGUAGE_NAMES.get(language, language)
            prompt = f"Translate this English text to {lang_name}. Reply with ONLY the translation:\n\n{text}"

        from groq import Groq
        r = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=800,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return text


# ── Chat Engine ────────────────────────────────────────────────────────────────

class ChatEngine:
    def __init__(self):
        self.status = ModuleStatus()
        self._lang:   Optional[LanguageDetector]  = None
        self._emotion: Optional[EmotionClassifier] = None
        self._intent:  Optional[IntentClassifier]  = None
        self._rag:     Optional[RAGPipeline]        = None
        self._groq_client = None

    def load(self) -> "ChatEngine":
        self._load_language()
        self._load_emotion()
        self._load_intent()
        self._load_rag()
        # Groq client for translation (reuse RAG client if available)
        if self._rag:
            self._groq_client = self._rag._groq
        return self

    def _load_language(self):
        try:
            self._lang = LanguageDetector()
            self.status.language = True
        except Exception as e:
            self.status.errors["language"] = str(e)

    def _load_emotion(self):
        try:
            self._emotion = EmotionClassifier()
            self.status.emotion = True
        except Exception as e:
            self.status.errors["emotion"] = str(e)

    def _load_intent(self):
        try:
            self._intent = IntentClassifier()
            self.status.intent = True
        except Exception as e:
            self.status.errors["intent"] = str(e)

    def _load_rag(self):
        try:
            self._rag = RAGPipeline()
            self.status.rag = True
        except Exception as e:
            self.status.errors["rag"] = str(e)

    # ── Main process ──────────────────────────────────────────────────────────

    def process(
        self,
        message:       str,
        session_id:    str  = "default",
        top_k:         int  = 5,
        strong_model:  bool = True,
    ) -> dict:
        """
        Run the full pipeline.

        Parameters
        ----------
        message      : user input text
        session_id   : identifies the conversation session
        top_k        : number of documents to retrieve from Qdrant (Module 4)
        strong_model : use gpt-oss-120b (True) or gpt-oss-20b (False)

        Returns
        -------
        dict with keys: response, language, language_name, lang_conf,
                        intent, intent_meta, emotion, emotion_meta,
                        sources, crisis, crisis_level, elapsed_ms
        """
        t_start = time.perf_counter()
        history = _sessions[session_id]
        message = message.strip()

        # ── 1. Crisis detection ────────────────────────────────────────────────
        crisis = detect_crisis(message)
        if crisis["level"] == "high":
            resp = crisis["response"]
            history.append(Turn("user",      message, {}))
            history.append(Turn("assistant", resp,    {"crisis": True, "crisis_level": "high"}))
            return self._result(
                message=message, response=resp,
                language="en", language_name="English", lang_conf=0.99,
                intent="asking_mental_health_question",
                emotion="fear", emotion_meta={},
                sources=[], crisis=True, crisis_level="high",
                elapsed=time.perf_counter() - t_start,
            )

        forced_tone = (
            "deeply compassionate and validating"
            if crisis["level"] == "medium" else None
        )

        # ── 2. Language detection ──────────────────────────────────────────────
        lang_result   = (
            self._lang.detect(message)
            if self._lang
            else {"language": "en", "language_name": "English", "confidence": 0.85}
        )
        language      = lang_result["language"]
        language_name = lang_result["language_name"]
        lang_conf     = lang_result.get("confidence", 0.85)

        # ── 3. Translate to English for NLP processing ────────────────────────
        en_message = (
            _translate(message, "to_en", language, self._groq_client)
            if language != "en" and self._groq_client
            else message
        )

        # ── 4. Intent classification ───────────────────────────────────────────
        intent_result = (
            self._intent.classify(en_message)
            if self._intent
            else {"intent": "asking_mental_health_question", "route": "rag",
                  "emoji": "🧠", "color": "#6C8EBF", "confidence": 0.5}
        )
        intent = intent_result["intent"]
        route  = intent_result["route"]

        # ── 5+6. Route ─────────────────────────────────────────────────────────
        if route == "rag" and self._rag:
            # Emotion classification
            emotion_result = (
                self._emotion.classify(en_message)
                if self._emotion
                else {"emotion": "neutral", "tone": "empathetic",
                      "emoji": "😐", "color": "#9E9E9E", "confidence": 0.0,
                      "all_scores": {}, "device": "cpu"}
            )
            emotion = emotion_result["emotion"]
            tone    = forced_tone or emotion_result["tone"]

            # Build conversation context for RAG (last user turn)
            conv_ctx = ""
            user_turns = [t.content for t in history if t.role == "user"]
            if user_turns:
                conv_ctx = user_turns[-1]

            rag_result = self._rag.answer(
                question=en_message,
                emotion=emotion,
                tone=tone,
                language=language,
                language_name=language_name,
                conversation_ctx=conv_ctx,
                top_k=top_k,
                use_strong_model=strong_model,
            )
            response_en = rag_result["response"]
            sources     = rag_result["sources"]

        else:
            emotion_result = {
                "emotion": None, "emoji": "", "color": "",
                "confidence": 0.0, "all_scores": {}, "device": "n/a",
            }
            emotion = None
            if self._rag:
                direct     = self._rag.direct_reply(en_message, intent=intent, language_name=language_name)
                response_en = direct["response"]
            else:
                response_en = self._fallback(intent)
            sources = []

        # ── 7. Translate response back if user wrote in non-English ────────────
        response = (
            _translate(response_en, "from_en", language, self._groq_client)
            if language != "en" and self._groq_client
            else response_en
        )

        # ── 8. Store session history ───────────────────────────────────────────
        history.append(Turn("user",      message,  {"language": language, "intent": intent}))
        history.append(Turn("assistant", response, {"emotion": emotion}))
        if len(history) > MAX_HISTORY * 2:
            _sessions[session_id] = history[-(MAX_HISTORY * 2):]

        return self._result(
            message=message, response=response,
            language=language, language_name=language_name, lang_conf=lang_conf,
            intent=intent, intent_meta=intent_result,
            emotion=emotion, emotion_meta=emotion_result,
            sources=sources, crisis=False,
            crisis_level=crisis["level"],
            elapsed=time.perf_counter() - t_start,
        )

    def get_history(self, session_id: str) -> list[Turn]:
        return _sessions.get(session_id, [])

    def clear_history(self, session_id: str) -> None:
        _sessions.pop(session_id, None)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(intent: str) -> str:
        return {
            "greeting":    "Hello! I'm your mental health support assistant. How can I help you today?",
            "goodbye":     "Take care of yourself. I'm always here if you need support. 💙",
            "gratitude":   "You're very welcome. I'm glad I could help.",
            "out_of_scope":"I'm best equipped to help with mental health topics like anxiety, depression, or stress. Is there something along those lines I can help with?",
        }.get(intent, "I'm here to support you. How can I help?")

    @staticmethod
    def _result(**kw) -> dict:
        return {
            "message":       kw.get("message", ""),
            "response":      kw.get("response", ""),
            "language":      kw.get("language", "en"),
            "language_name": kw.get("language_name", "English"),
            "intent":        kw.get("intent", ""),
            "intent_meta":   kw.get("intent_meta", {}),
            "emotion":       kw.get("emotion"),
            "emotion_meta":  kw.get("emotion_meta", {}),
            "sources":       kw.get("sources", []),
            "crisis":        kw.get("crisis", False),
            "crisis_level":  kw.get("crisis_level", "low"),
            "elapsed_ms":    round(kw.get("elapsed", 0) * 1000, 1),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine_instance: Optional[ChatEngine] = None


def get_engine() -> ChatEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatEngine().load()
    return _engine_instance
