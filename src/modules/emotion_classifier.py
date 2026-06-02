"""
Module 2 — Emotion Classifier
DistilBERT fine-tuned on dair-ai/emotion (6 classes).
Device priority: MPS (Apple Silicon) → CUDA → CPU.
"""
from pathlib import Path
import numpy as np
import torch

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "emotion_distilbert"

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_META = {
    "sadness":  {"emoji": "😢", "color": "#6C8EBF", "tone": "deeply compassionate and validating"},
    "joy":      {"emoji": "😊", "color": "#82B366", "tone": "warm, positive, and encouraging"},
    "love":     {"emoji": "💙", "color": "#D6A4C7", "tone": "warm and genuinely affirming"},
    "anger":    {"emoji": "😤", "color": "#D79B00", "tone": "calm, empathetic, and de-escalating"},
    "fear":     {"emoji": "😰", "color": "#67AB9F", "tone": "reassuring, grounding, and steady"},
    "surprise": {"emoji": "😮", "color": "#9673A6", "tone": "clarifying, gentle, and supportive"},
}


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EmotionClassifier:
    def __init__(self, model_path: str | None = None):
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._device = _best_device()
        if path.exists():
            model_source = str(path)
        else:
            # Fallback to pre-trained Hub model trained on the same dair-ai/emotion dataset
            model_source = "bhadresh-savani/distilbert-base-uncased-emotion"

        self._tokenizer = DistilBertTokenizerFast.from_pretrained(model_source)
        self._model     = DistilBertForSequenceClassification.from_pretrained(model_source)
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def classify(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return self._build_result("sadness", np.zeros(len(EMOTION_LABELS)))

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=128, padding="max_length",
        )
        # Move each tensor to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        logits = self._model(**inputs).logits[0]
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        idx    = int(np.argmax(probs))
        return self._build_result(EMOTION_LABELS[idx], probs)

    def _build_result(self, label: str, probs: np.ndarray) -> dict:
        scores = {EMOTION_LABELS[i]: round(float(probs[i]), 4) for i in range(len(EMOTION_LABELS))}
        meta   = EMOTION_META[label]
        return {
            "emotion":    label,
            "emoji":      meta["emoji"],
            "color":      meta["color"],
            "tone":       meta["tone"],
            "confidence": round(float(max(probs)) if probs.any() else 0.0, 4),
            "all_scores": scores,
            "device":     str(self._device),
        }

    def is_ready(self) -> bool:
        return True


def get_classifier(model_path: str | None = None) -> EmotionClassifier:
    return EmotionClassifier(model_path)
