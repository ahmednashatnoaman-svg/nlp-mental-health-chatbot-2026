"""
Module 3 — Intent Classifier
Few-shot LLM prompting via Groq API (gpt-oss-20b — fastest model).
No training required.
"""
import json
import os
import re
from groq import Groq

INTENTS = ["greeting", "goodbye", "gratitude", "asking_mental_health_question", "out_of_scope"]

INTENT_META = {
    "greeting":                    {"emoji": "👋", "color": "#82B366", "route": "direct"},
    "goodbye":                     {"emoji": "👋", "color": "#9673A6", "route": "direct"},
    "gratitude":                   {"emoji": "🙏", "color": "#67AB9F", "route": "direct"},
    "asking_mental_health_question": {"emoji": "🧠", "color": "#6C8EBF", "route": "rag"},
    "out_of_scope":                {"emoji": "🚫", "color": "#D79B00", "route": "direct"},
}

_SYSTEM_PROMPT = """You are an intent classifier for a mental health support chatbot.
Classify the user's message into EXACTLY ONE of these intents:
- greeting
- goodbye
- gratitude
- asking_mental_health_question
- out_of_scope

RULES:
- "asking_mental_health_question": ANY expression of emotion or feelings — positive OR negative.
  This includes: happiness, excitement, sadness, anxiety, anger, loneliness, confusion,
  stress, depression, panic, love, fear, hope, frustration, boredom, or any other emotional state.
  Also includes questions about therapy, coping, sleep, relationships, self-worth, purpose,
  mental wellbeing, or personal growth.
  KEY: If the user is sharing HOW THEY FEEL in any way, route to asking_mental_health_question.
- "greeting": simple hello/hi/good morning with NO emotional content.
- "goodbye": farewell messages with NO emotional content.
- "gratitude": thank-you messages with NO additional emotional sharing.
- "out_of_scope": ONLY for topics with zero emotional content — weather forecasts, math problems,
  sports scores, coding tasks, geography questions, factual encyclopedic queries.
  IMPORTANT: if there is ANY hint of emotion or personal feeling, do NOT classify as out_of_scope.
- Respond with ONLY valid JSON: {"intent": "<class>", "confidence": <0.0-1.0>}

FEW-SHOT EXAMPLES:
User: "Hello there!" → {"intent": "greeting", "confidence": 0.99}
User: "Hi, how are you?" → {"intent": "greeting", "confidence": 0.97}
User: "Good morning!" → {"intent": "greeting", "confidence": 0.98}
User: "Bye, see you later" → {"intent": "goodbye", "confidence": 0.99}
User: "Goodbye, take care" → {"intent": "goodbye", "confidence": 0.98}
User: "Thank you so much, that really helped" → {"intent": "gratitude", "confidence": 0.99}
User: "I appreciate your support" → {"intent": "gratitude", "confidence": 0.97}
User: "I am very happy today" → {"intent": "asking_mental_health_question", "confidence": 0.92}
User: "I feel great and energized" → {"intent": "asking_mental_health_question", "confidence": 0.90}
User: "I'm so excited about life right now" → {"intent": "asking_mental_health_question", "confidence": 0.91}
User: "I had a wonderful day" → {"intent": "asking_mental_health_question", "confidence": 0.88}
User: "I feel sad today" → {"intent": "asking_mental_health_question", "confidence": 0.97}
User: "I feel confused about my life" → {"intent": "asking_mental_health_question", "confidence": 0.95}
User: "I've been feeling very anxious lately, what should I do?" → {"intent": "asking_mental_health_question", "confidence": 0.99}
User: "I can't sleep because of panic attacks" → {"intent": "asking_mental_health_question", "confidence": 0.98}
User: "How do I cope with depression?" → {"intent": "asking_mental_health_question", "confidence": 0.99}
User: "I feel like I'm not good enough" → {"intent": "asking_mental_health_question", "confidence": 0.96}
User: "I am lonely" → {"intent": "asking_mental_health_question", "confidence": 0.97}
User: "I feel empty inside" → {"intent": "asking_mental_health_question", "confidence": 0.98}
User: "I'm stressed about work" → {"intent": "asking_mental_health_question", "confidence": 0.97}
User: "What's the weather like today?" → {"intent": "out_of_scope", "confidence": 0.99}
User: "Can you write me some Python code?" → {"intent": "out_of_scope", "confidence": 0.99}
User: "What is the capital of France?" → {"intent": "out_of_scope", "confidence": 0.99}
User: "What is 2 + 2?" → {"intent": "out_of_scope", "confidence": 0.99}"""


class IntentClassifier:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("LLM_API_KEY is not set. Add it to your .env file.")
        self._client = Groq(api_key=key)

    def classify(self, text: str) -> dict:
        """Return intent, confidence, emoji, route (rag vs direct)."""
        text = text.strip()
        if not text:
            return self._build_result("out_of_scope", 0.5)

        response = self._client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=100,  # 'asking_mental_health_question' needs ~60 chars minimum
        )

        raw = response.choices[0].message.content.strip()
        return self._parse(raw)

    def _parse(self, raw: str) -> dict:
        if not raw:
            return self._build_result("out_of_scope", 0.5)

        # Try complete JSON first
        try:
            match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                intent = data.get("intent", "out_of_scope")
                if intent not in INTENTS:
                    intent = "out_of_scope"
                return self._build_result(intent, float(data.get("confidence", 0.8)))
        except Exception:
            pass

        # Fallback: extract intent keyword from partial/truncated JSON
        for intent in INTENTS:
            if intent in raw:
                return self._build_result(intent, 0.8)

        return self._build_result("out_of_scope", 0.5)

    def _build_result(self, intent: str, confidence: float) -> dict:
        meta = INTENT_META.get(intent, INTENT_META["out_of_scope"])
        return {
            "intent": intent,
            "emoji": meta["emoji"],
            "color": meta["color"],
            "route": meta["route"],
            "confidence": round(confidence, 4),
        }

    def is_ready(self) -> bool:
        return True


def get_classifier(api_key: str | None = None) -> IntentClassifier:
    return IntentClassifier(api_key)
