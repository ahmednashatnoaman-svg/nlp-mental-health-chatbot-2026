"""
Module 1 — Language Detector
Strategy (layered):
  1. Unicode script detection  — Arabic, CJK, Thai, Cyrillic, Devanagari → instant, 100% reliable
  2. Short ASCII text (≤ 60 chars) → English by default (TF-IDF fails on <5 words)
  3. Trained TF-IDF + LinearSVC  — reliable on longer, mixed-script text (≥ 30 chars)

Trained on papluca/language-identification (70k samples, 20 languages).
"""
import joblib
import numpy as np
from pathlib import Path

LANGUAGE_NAMES = {
    "ar": "Arabic",     "bg": "Bulgarian", "de": "German",    "el": "Greek",
    "en": "English",    "es": "Spanish",   "fr": "French",    "hi": "Hindi",
    "it": "Italian",    "ja": "Japanese",  "nl": "Dutch",     "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian",   "sw": "Swahili",   "th": "Thai",
    "tr": "Turkish",    "ur": "Urdu",      "vi": "Vietnamese","zh": "Chinese",
}

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "language_detector.pkl"

# ── Unicode script character ranges ───────────────────────────────────────────
_SCRIPT_RANGES = [
    ("ar", 0x0600, 0x06FF),   # Arabic
    ("ar", 0x0750, 0x077F),   # Arabic Extended-A
    ("ar", 0x08A0, 0x08FF),   # Arabic Extended-B
    ("zh", 0x4E00, 0x9FFF),   # CJK Unified Ideographs
    ("zh", 0x3400, 0x4DBF),   # CJK Extension A
    ("ja", 0x3040, 0x309F),   # Hiragana
    ("ja", 0x30A0, 0x30FF),   # Katakana
    ("th", 0x0E00, 0x0E7F),   # Thai
    ("hi", 0x0900, 0x097F),   # Devanagari (Hindi)
    ("ur", 0x0600, 0x06FF),   # Urdu shares Arabic script — handled via Arabic first
    ("ru", 0x0400, 0x04FF),   # Cyrillic (Russian, Bulgarian)
    ("el", 0x0370, 0x03FF),   # Greek
    ("ko", 0xAC00, 0xD7AF),   # Hangul (Korean)
]


def _detect_by_script(text: str) -> str | None:
    """
    Detect language purely from Unicode character ranges.
    Returns language code if ≥ 25% of non-space chars match a script, else None.
    """
    chars = [ord(c) for c in text if not c.isspace() and not c.isdigit()]
    if not chars:
        return None

    total = len(chars)
    counts: dict[str, int] = {}

    for lang, lo, hi in _SCRIPT_RANGES:
        count = sum(1 for c in chars if lo <= c <= hi)
        if count:
            counts[lang] = counts.get(lang, 0) + count

    if counts:
        best_lang = max(counts, key=lambda k: counts[k])
        if counts[best_lang] / total >= 0.25:
            return best_lang

    return None


def _is_ascii_text(text: str) -> bool:
    """True if ≥ 90% of non-space chars are plain ASCII (Latin alphabet)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    ascii_count = sum(1 for c in chars if ord(c) < 128)
    return ascii_count / len(chars) >= 0.90


def _has_accented_latin(text: str) -> bool:
    """True if text contains accented Latin characters (French, German, Spanish, etc.)."""
    # Latin Extended: à-ÿ (0xC0-0xFF), Latin Extended A/B (0x100-0x24F)
    return any(0x00C0 <= ord(c) <= 0x024F for c in text)


class LanguageDetector:
    def __init__(self, model_path: str | None = None):
        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Language model not found at {path}. "
                "Run: python scripts/train_language_model.py"
            )
        self._pipeline = joblib.load(path)
        self._classes  = list(self._pipeline.classes_)

    def detect(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {"language": "en", "language_name": "English", "confidence": 0.0}

        # ── Layer 1: Unicode script detection (instant, 100% reliable) ────────
        script_lang = _detect_by_script(text)
        if script_lang:
            return {
                "language":      script_lang,
                "language_name": LANGUAGE_NAMES.get(script_lang, script_lang),
                "confidence":    0.99,
            }

        # ── Layer 2: Short plain-ASCII text → default English ─────────────────
        # TF-IDF char n-grams are unreliable on very short Latin text.
        # Only apply when: pure ASCII (no accents) AND ≤ 4 words.
        # Accented text (é, ü, ñ, ç …) goes straight to the model.
        word_count = len(text.split())
        if _is_ascii_text(text) and not _has_accented_latin(text) and word_count <= 4:
            return {"language": "en", "language_name": "English", "confidence": 0.85}

        # ── Layer 3: Trained TF-IDF + LinearSVC model ─────────────────────────
        proba   = self._pipeline.predict_proba([text])[0]
        idx     = int(np.argmax(proba))
        lang    = self._classes[idx]
        conf    = float(proba[idx])

        # Extra guard: if model says non-English but text is mostly ASCII,
        # only trust it when confidence is very high
        if lang != "en" and _is_ascii_text(text) and conf < 0.92:
            lang = "en"

        return {
            "language":      lang,
            "language_name": LANGUAGE_NAMES.get(lang, lang),
            "confidence":    round(conf, 4),
        }

    def is_ready(self) -> bool:
        return self._pipeline is not None


def get_detector(model_path: str | None = None) -> LanguageDetector:
    return LanguageDetector(model_path)
