"""
Crisis Detection Layer — Multi-level risk scoring: HIGH / MEDIUM / LOW
Runs BEFORE the main pipeline and overrides routing on HIGH risk.

After the pipeline reorder (v2), crisis detection runs on the English-translated
text, so non-English messages are properly caught. Multilingual keyword banks
below serve as a belt-and-suspenders safety net in case translation is unavailable.
"""
import re

# ── English HIGH-risk phrases ─────────────────────────────────────────────────

HIGH_RISK_PHRASES = [
    "suicide", "suicidal", "kill myself", "end my life", "want to die",
    "don't want to live", "no reason to live", "better off dead",
    "can't go on", "cannot go on", "give up on life", "take my own life",
    "end it all", "not worth living", "life is not worth",
    "ending my life", "take my life",
    "self-harm", "self harm", "cutting myself", "hurt myself",
    "overdose", "take too many pills", "hanging myself", "jump off",
    "no point in living", "everyone would be better without me",
    "don't want to be here anymore", "cannot be here anymore",
    "i want to disappear forever", "wish i was dead",
]

# ── Arabic HIGH-risk phrases ───────────────────────────────────────────────────

HIGH_RISK_AR = [
    "أريد الانتحار",        # I want to commit suicide
    "سأنتحر",               # I will commit suicide
    "أريد أن أموت",         # I want to die
    "لا أريد أن أعيش",      # I don't want to live
    "سأقتل نفسي",           # I will kill myself
    "أريد قتل نفسي",        # I want to kill myself
    "إنهاء حياتي",          # ending my life
    "أذية نفسي",            # hurting myself
    "جرح نفسي",             # cutting myself
    "الانتحار",              # suicide (standalone)
    "لا قيمة للحياة",        # life has no value
    "أتمنى لو مت",          # I wish I had died
    "الموت أفضل",           # death is better
]

# ── Arabic MEDIUM-risk phrases ─────────────────────────────────────────────────

MEDIUM_RISK_AR = [
    "لا أحد يهتم",          # nobody cares
    "لا أحد يفهمني",        # nobody understands me
    "أشعر باليأس",          # I feel hopeless
    "يائس تماماً",          # completely hopeless
    "أشعر بعدم الجدوى",     # feeling worthless
    "لا معنى للحياة",        # life has no meaning
    "تعبت من الحياة",        # tired of life
    "لا طريق للخروج",        # no way out
    "عالق",                  # trapped
    "أشعر بالفراغ",          # feel empty
    "محبط جداً",            # very frustrated / despairing
]

# ── French HIGH-risk phrases ───────────────────────────────────────────────────

HIGH_RISK_FR = [
    "je veux mourir",        # I want to die
    "me suicider",           # commit suicide (myself)
    "mettre fin à ma vie",   # end my life
    "en finir avec la vie",  # be done with life
    "me tuer",               # kill myself
    "je veux me tuer",       # I want to kill myself
    "me faire du mal",       # harm myself
]

# ── Spanish HIGH-risk phrases ──────────────────────────────────────────────────

HIGH_RISK_ES = [
    "quiero morir",          # I want to die
    "quiero suicidarme",     # I want to commit suicide
    "quitarme la vida",      # take my life
    "hacerme daño",          # hurt myself
    "no quiero vivir",       # I don't want to live
    "acabar con todo",       # end everything
]

# ── English MEDIUM-risk phrases ────────────────────────────────────────────────

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

# ── False-positive exclusions ─────────────────────────────────────────────────

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

def _score(text: str) -> int:
    """
    Returns a risk score:
      10+ = HIGH  (override with crisis resources)
       5-9 = MEDIUM (empathetic re-route, no RAG skip)
       0-4 = LOW   (normal pipeline)

    Checks English phrases first; also checks multilingual phrase banks as a
    safety net when the text has not been translated yet.
    """
    text_lower = text.lower()

    # Exclude known false positives (English only)
    for excl in _EXCLUSION_PHRASES:
        if excl in text_lower:
            return 0

    score = 0

    # English HIGH-risk
    for phrase in HIGH_RISK_PHRASES:
        if phrase in text_lower:
            score += 10
            break

    # Multilingual HIGH-risk (Arabic, French, Spanish)
    if score < 10:
        for phrase in (*HIGH_RISK_AR, *HIGH_RISK_FR, *HIGH_RISK_ES):
            if phrase in text:   # case-sensitive for Arabic/accented Latin
                score += 10
                break

    # English MEDIUM-risk
    if score < 5:
        for phrase in MEDIUM_RISK_PHRASES:
            if phrase in text_lower:
                score += 5
                break

    # Arabic MEDIUM-risk
    if score < 5:
        for phrase in MEDIUM_RISK_AR:
            if phrase in text:
                score += 5
                break

    return score


def detect_crisis(text: str) -> dict:
    """
    Returns:
      {"level": "high"|"medium"|"low", "response": str|None, "score": int}
    """
    score = _score(text)

    if score >= 10:
        return {"level": "high",   "response": CRISIS_RESOURCES,    "score": score}
    if score >= 5:
        return {"level": "medium", "response": MEDIUM_RISK_RESPONSE, "score": score}
    return     {"level": "low",    "response": None,                 "score": score}


def is_crisis(text: str) -> bool:
    """Legacy alias used in tests."""
    return detect_crisis(text)["level"] == "high"


def get_crisis_response() -> str:
    return CRISIS_RESOURCES
