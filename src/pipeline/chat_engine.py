"""
Chat Engine — Orchestrates all 4 NLP modules end-to-end.

Pipeline (revised order for correct multilingual crisis detection):
  1. Language detection  (Module 1)         — fast, mostly regex/rule-based
  2. Translate to English if needed         — Groq gpt-oss-20b
  3. Crisis detection                        — on English text, catches all languages
  4. Intent classification (Module 3)       — few-shot LLM
  5. Emotion classification (Module 2)      — DistilBERT, only for mental-health queries
  6. RAG answer OR direct LLM reply (Module 4)
  7. (If HIGH crisis AND non-English) translate crisis response back

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
            prompt = (
                f"Translate this {language} text to English. "
                f"Reply with ONLY the translation, no explanations:\n\n{text}"
            )
        else:
            from src.modules.language_detector import LANGUAGE_NAMES
            lang_name = LANGUAGE_NAMES.get(language, language)
            prompt = (
                f"Translate this English text to {lang_name}. "
                f"Reply with ONLY the translation, no explanations:\n\n{text}"
            )

        from groq import Groq
        r = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=800,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return text


# ── Crisis merge helper ────────────────────────────────────────────────────────

def _merge_crisis(a: dict, b: dict) -> dict:
    """Return the more severe crisis result of two detect_crisis() outputs."""
    rank = {"high": 2, "medium": 1, "low": 0}
    if rank[a["level"]] >= rank[b["level"]]:
        return a
    return b


# ── Chat Engine ────────────────────────────────────────────────────────────────

class ChatEngine:
    def __init__(self):
        self.status = ModuleStatus()
        self._lang:    Optional[LanguageDetector]  = None
        self._emotion: Optional[EmotionClassifier] = None
        self._intent:  Optional[IntentClassifier]  = None
        self._rag:     Optional[RAGPipeline]        = None
        self._groq_client = None

    def load(self) -> "ChatEngine":
        self._load_language()
        self._load_emotion()
        self._load_intent()
        self._load_rag()
        # Groq client for translation — reuse RAG client to avoid extra init
        if self._rag:
            self._groq_client = self._rag._groq
        elif self._intent:
            self._groq_client = self._intent._client
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

        Pipeline order (revised):
          1. Language detection  → know the language before crisis check
          2. Translate → English  → crisis keywords are English, detect on en_message
          3. Crisis detection    → on English representation (catches all languages)
          4. Intent classification
          5. Emotion + RAG  (mental health path)  OR  direct LLM reply
          6. Return result with all metadata

        Parameters
        ----------
        message      : raw user input text (any language)
        session_id   : conversation identifier
        top_k        : passages retrieved from Qdrant (Module 4)
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

        # ── 1. Language detection ──────────────────────────────────────────────
        lang_result = (
            self._lang.detect(message)
            if self._lang
            else {"language": "en", "language_name": "English", "confidence": 0.85}
        )
        language      = lang_result["language"]
        language_name = lang_result["language_name"]
        lang_conf     = lang_result.get("confidence", 0.85)

        # ── 2. Translate to English for NLP processing ────────────────────────
        # Crisis, intent, and emotion models are English-based.
        en_message = (
            _translate(message, "to_en", language, self._groq_client)
            if language != "en" and self._groq_client
            else message
        )

        # ── 3. Crisis detection — dual-pass for maximum safety ────────────────
        # Pass 1: original message — catches multilingual keywords (AR/FR/ES/EN)
        # Pass 2: translated text  — catches English keywords in translated text
        # Take the most severe result of the two passes.
        crisis_orig = detect_crisis(message)
        crisis_en   = detect_crisis(en_message) if en_message != message else crisis_orig
        crisis = _merge_crisis(crisis_orig, crisis_en)

        if crisis["level"] == "high":
            resp_en = crisis["response"]
            # Translate crisis response back to user's language if non-English
            resp = (
                _translate(resp_en, "from_en", language, self._groq_client)
                if language != "en" and self._groq_client
                else resp_en
            )
            history.append(Turn("user",      message, {}))
            history.append(Turn("assistant", resp,    {"crisis": True, "crisis_level": "high"}))
            return self._result(
                message=message, response=resp,
                language=language, language_name=language_name, lang_conf=lang_conf,
                intent="asking_mental_health_question",
                emotion="fear", emotion_meta={},
                sources=[], crisis=True, crisis_level="high",
                elapsed=time.perf_counter() - t_start,
            )

        forced_tone = (
            "deeply compassionate and validating"
            if crisis["level"] == "medium" else None
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

        # ── 5+6. Route by intent ───────────────────────────────────────────────
        if route == "rag" and self._rag:
            # 5. Emotion classification
            emotion_result = (
                self._emotion.classify(en_message)
                if self._emotion
                else {"emotion": "neutral", "tone": "empathetic",
                      "emoji": "😐", "color": "#9E9E9E", "confidence": 0.0,
                      "all_scores": {}, "device": "cpu"}
            )
            emotion = emotion_result["emotion"]
            tone    = forced_tone or emotion_result["tone"]

            # Build conversation context for retrieval (last user turn for context)
            conv_ctx = ""
            user_turns = [t.content for t in history if t.role == "user"]
            if user_turns:
                conv_ctx = user_turns[-1]

            # 6. RAG generation (LLM prompted to respond in language_name)
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
            response = rag_result["response"]
            sources  = rag_result["sources"]

        else:
            # Direct LLM reply (greeting, goodbye, gratitude, out-of-scope)
            emotion_result = {
                "emotion": None, "emoji": "", "color": "",
                "confidence": 0.0, "all_scores": {}, "device": "n/a",
            }
            emotion = None

            if self._rag:
                # LLM direct reply — prompted to respond in language_name
                direct   = self._rag.direct_reply(en_message, intent=intent,
                                                   language_name=language_name)
                response = direct["response"]
            else:
                # Hard fallback when RAG module unavailable
                response_en = self._fallback(intent)
                response = (
                    _translate(response_en, "from_en", language, self._groq_client)
                    if language != "en" and self._groq_client
                    else response_en
                )
            sources = []

        # ── 7. Store session history ───────────────────────────────────────────
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
            "out_of_scope": "I'm best equipped to help with mental health topics like anxiety, depression, or stress. Is there something along those lines I can help with?",
        }.get(intent, "I'm here to support you. How can I help?")

    @staticmethod
    def _result(**kw) -> dict:
        return {
            "message":       kw.get("message", ""),
            "response":      kw.get("response", ""),
            "language":      kw.get("language", "en"),
            "language_name": kw.get("language_name", "English"),
            "lang_conf":     kw.get("lang_conf", 0.85),
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
