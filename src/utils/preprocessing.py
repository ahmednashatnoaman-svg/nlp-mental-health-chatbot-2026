"""Shared text-cleaning utilities used across all modules."""
import re


def clean_text(text: str, remove_urls: bool = True, remove_mentions: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
    if remove_mentions:
        text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_language_detection(text: str) -> str:
    """Minimal cleaning — preserve script characters (Arabic, Chinese, etc.)."""
    return re.sub(r"\s+", " ", text.strip())


def clean_for_emotion(text: str) -> str:
    """Clean for emotion model — keep emotional punctuation."""
    text = clean_text(text, remove_urls=True, remove_mentions=True)
    text = re.sub(r"[^\w\s!?.,;:'\"()-]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_qa_chunk(context: str, response: str) -> str:
    """Format a counseling Q&A pair as a single chunk for RAG ingestion."""
    ctx = context.strip()
    resp = response.strip()
    return f"Patient: {ctx}\nCounselor: {resp}"
