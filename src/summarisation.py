# src/summarisation.py
from transformers import pipeline

_summariser = pipeline("summarization", model="facebook/bart-base")

def naive_summary(text: str, max_length: int = 120, min_length: int = 50) -> str:
    out = _summariser(text, max_length=max_length, min_length=min_length, do_sample=False)
    return out[0]["summary_text"].strip()

def emotion_aware_summary(text: str, max_length: int = 120, min_length: int = 50) -> str:
    """
    Minimal 'emotion-aware' variant for BART (non-instructional):
    - Create base summary
    - Append 1 emotional sentence from the original story if missing
    This creates a measurable difference without LLM prompting.
    """
    base = naive_summary(text, max_length=max_length, min_length=min_length)

    # Simple heuristic: keep one sentence that contains emotion words
    emotion_cues = ["happy", "sad", "scared", "afraid", "proud", "pain", "joy", "angry", "excited", "cry", "love"]
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

    chosen = ""
    for s in sentences:
        low = s.lower()
        if any(w in low for w in emotion_cues):
            chosen = s
            break

    # If base doesn't already contain that emotional sentence, append it.
    if chosen and chosen.lower() not in base.lower():
        return (base + ". " + chosen + ".").strip()

    return base

#issuecommit*/