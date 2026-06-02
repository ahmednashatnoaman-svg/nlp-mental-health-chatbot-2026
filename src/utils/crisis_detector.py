"""
Bonus — Crisis Detection Layer  (upgraded from mental-health-analyzer skill)
Multi-level risk scoring: HIGH / MEDIUM / LOW
Always runs BEFORE the main pipeline and overrides routing on HIGH risk.
"""
import re

# ── Risk-level keyword banks ──────────────────────────────────────────────────

HIGH_RISK_PHRASES = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "don't want to live", "no reason to live", "better off dead",
    "can't go on", "cannot go on", "give up on life", "take my own life",
    "end it all", "not worth living", "life is not worth",
    "ending my life", "end my life", "take my life",
    "self-harm", "self harm", "cutting myself", "hurt myself",
    "overdose", "take too many pills", "hanging myself", "jump off",
    "no point in living", "everyone would be better without me",
    "don't want to be here anymore", "cannot be here anymore",
    "i want to disappear forever", "wish i was dead",
]

MEDIUM_RISK_PHRASES = [
    "hopeless", "feel hopeless", "completely hopeless", "totally hopeless",
    "worthless", "feel worthless",
    "nobody cares", "no one cares", "nobody cares about me",
    "all alone", "completely alone", "totally alone",
    "can't take it anymore", "cannot take this anymore", "can't take this",
    "i give up", "giving up on everything",
    "nothing to live for", "trapped", "no way out", "pointless",
    "hate myself", "wish i wasn't here", "wish i was never born",
    "numb inside", "empty inside", "feel empty",
    "exhausted with life", "tired of living",
    "no one understands", "nobody understands me",
]

# Phrases that are NOT crisis when standing alone (false-positive guards)
_EXCLUSION_PHRASES = [
    "dying of laughter", "killing it", "dead tired", "drop dead gorgeous",
    "die hard", "die laughing",
]

# ── Crisis resources ──────────────────────────────────────────────────────────

CRISIS_RESOURCES = """I'm deeply concerned about what you've shared, and I want you to know that **you matter** and you are not alone in this.

**Please reach out to a crisis support line right now — they are available 24/7:**

🆘 **International Crisis Lines:**
- 🌍 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
- 🌍 Befrienders Worldwide: https://www.befrienders.org

📞 **By Region:**
- 🇺🇸 USA — 988 Suicide & Crisis Lifeline: Call or text **988**
- 🇬🇧 UK — Samaritans: **116 123**
- 🇦🇺 AU — Lifeline: **13 11 14**
- 🇨🇦 CA — Crisis Services Canada: **1-833-456-4566**
- 🌐 Crisis Text Line: Text **HOME** to **741741**

You don't have to face this alone. A trained counselor is ready to listen right now.
I'm here too — please tell me more about what you're going through."""

MEDIUM_RISK_RESPONSE = """What you're sharing sounds really painful, and I want you to know I'm here and I'm listening.

These feelings are valid, and it's okay to not be okay sometimes. You reached out, and that takes courage.

Can you tell me more about what's been going on? I'd like to understand what you're experiencing."""


# ── Detection logic ───────────────────────────────────────────────────────────

def _score(text_lower: str) -> int:
    """
    Returns a risk score:
      10+ = HIGH (override with crisis resources)
       5-9 = MEDIUM (empathetic re-route, no RAG skip)
       0-4 = LOW (normal pipeline)
    """
    # Exclude known false positives
    for excl in _EXCLUSION_PHRASES:
        if excl in text_lower:
            return 0

    score = 0
    for phrase in HIGH_RISK_PHRASES:
        if phrase in text_lower:
            score += 10
            break  # one high-risk phrase is enough

    for phrase in MEDIUM_RISK_PHRASES:
        if phrase in text_lower:
            score += 5  # single medium phrase → score 5 → MEDIUM level
            break       # one match is sufficient

    return score


def detect_crisis(text: str) -> dict:
    """
    Returns:
      {"level": "high"|"medium"|"low", "response": str|None}
    """
    text_lower = text.lower()
    score = _score(text_lower)

    if score >= 10:
        return {"level": "high",   "response": CRISIS_RESOURCES,      "score": score}
    if score >= 5:
        return {"level": "medium", "response": MEDIUM_RISK_RESPONSE,   "score": score}
    return     {"level": "low",    "response": None,                   "score": score}


# Legacy alias used by chat_engine
def is_crisis(text: str) -> bool:
    return detect_crisis(text)["level"] == "high"


def get_crisis_response() -> str:
    return CRISIS_RESOURCES
