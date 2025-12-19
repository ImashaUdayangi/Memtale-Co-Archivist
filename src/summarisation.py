# src/summarisation.py
from transformers import pipeline

_summariser = pipeline(
    "summarization",
    model="facebook/bart-base",
    tokenizer="facebook/bart-base",
)

# Force tokenizer max length for safe truncation behavior
_summariser.tokenizer.model_max_length = 1024

# BART has a max input length (~1024 tokens). StoryCorps transcripts can exceed that.
# Truncating prevents: IndexError: index out of range in self
def _truncate_chars(text: str, max_chars: int = 3500) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:max_chars]

def naive_summary(text: str, max_length: int = 120, min_length: int = 50) -> str:
    text = _truncate_chars(text, max_chars=3500)
    out = _summariser(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True
    )
    return out[0]["summary_text"].strip()

def emotion_aware_summary(text: str, max_length: int = 120, min_length: int = 50) -> str:
    """
    Minimal 'emotion-aware' variant for BART:
    - Create base summary (on truncated text)
    - Append 1 emotional sentence from the original text if missing
    """
    text_clean = _truncate_chars(text, max_chars=3500)
    base = naive_summary(text_clean, max_length=max_length, min_length=min_length)

    # Simple heuristic: keep one sentence that contains emotion words
    emotion_cues = ["happy", "sad", "scared", "afraid", "proud", "pain", "joy", "angry", "excited", "cry", "love"]

    # Use the same truncated version here so sentence splitting doesn't scan huge text
    sentences = [s.strip() for s in text_clean.split(".") if s.strip()]

    chosen = ""
    for s in sentences:
        low = s.lower()
        if any(w in low for w in emotion_cues):
            chosen = s
            break

    # If base doesn't already contain that emotional sentence, append it.
    if chosen and chosen.lower() not in base.lower():
        return (base.rstrip(".") + ". " + chosen.rstrip(".") + ".").strip()

    return base
