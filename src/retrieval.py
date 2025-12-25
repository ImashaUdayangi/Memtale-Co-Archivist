# src/retrieval.py
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# SBERT model (good default)
_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# B1: Keyword retrieval (BM25)
# -------------------------
def _tokenize(text: str):
    # simple word tokenizer (fast + stable)
    return re.findall(r"\b\w+\b", (text or "").lower())

def bm25_retrieval(query: str, docs: list[str], top_k: int = 5):
    tokenized_docs = [_tokenize(d) for d in docs]
    tokenized_docs = [toks if toks else ["_empty_"] for toks in tokenized_docs] 
    bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)

    scores = bm25.get_scores(_tokenize(query))  # numpy array
    ranked = np.argsort(scores)[::-1][:top_k]
    return ranked.tolist(), scores[ranked].tolist()

# -------------------------
# B2/B3: Semantic retrieval (SBERT)
# -------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    return _sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def semantic_retrieval(query: str, docs: list[str], top_k: int = 5):
    doc_embs = embed_texts(docs)
    q_emb = embed_texts([query])[0]
    sims = doc_embs @ q_emb  # cosine sim because normalized
    ranked = np.argsort(sims)[::-1][:top_k]
    return ranked.tolist(), sims[ranked].tolist()

# -------------------------
# Ours: Semantic + Emotion hybrid retrieval
# -------------------------
def emotion_aware_retrieval(
    query: str,
    docs: list[str],
    doc_emotions: list[dict],
    query_emotion: dict,
    alpha: float = 0.7,
    top_k: int = 5
):
    """
    Score = alpha * semantic_sim + (1-alpha) * emotion_sim
    emotion_sim computed as cosine similarity over emotion vectors.
    """
    # semantic sims
    doc_embs = embed_texts(docs)
    q_emb = embed_texts([query])[0]
    sem_sims = doc_embs @ q_emb

    # emotion sims
    keys = sorted(query_emotion.keys())
    q_e = np.array([query_emotion[k] for k in keys], dtype=float)
    q_norm = np.linalg.norm(q_e)

    # FALLBACK: if query has no meaningful emotion, use semantic only
    if q_norm < 1e-6:
        ranked = np.argsort(sem_sims)[::-1][:top_k]
        return ranked.tolist(), sem_sims[ranked].tolist(), sem_sims[ranked].tolist(), None

    # normalize query emotion
    q_e = q_e / q_norm


    emo_sims = []
    for e in doc_emotions:
        d_e = np.array([e[k] for k in keys], dtype=float)
        d_e = d_e / (np.linalg.norm(d_e) + 1e-12)
        emo_sims.append(float(d_e @ q_e))

    emo_sims = np.array(emo_sims)

    final = alpha * sem_sims + (1 - alpha) * emo_sims
    ranked = np.argsort(final)[::-1][:top_k]
    return ranked.tolist(), final[ranked].tolist(), sem_sims[ranked].tolist(), emo_sims[ranked].tolist()

# -------------------------
# Helpers for label-based emotion vectors (32 ED labels)
# -------------------------
def build_label_space(labels: list[str]):
    labels = sorted(set(str(x).lower().strip() for x in labels if str(x).strip()))
    idx = {lab: i for i, lab in enumerate(labels)}
    return labels, idx

def onehot(label: str, idx: dict):
    v = np.zeros(len(idx), dtype=float)
    lab = str(label).lower().strip()
    if lab in idx:
        v[idx[lab]] = 1.0
    return v

# -------------------------
# Ours (NEW): Semantic + Label-Emotion hybrid
# -------------------------
def emotion_aware_retrieval_vec(
    query: str,
    docs: list[str],
    doc_emo_vecs: np.ndarray,
    q_emo_vec: np.ndarray | None,
    alpha: float = 0.7,
    top_k: int = 5
):
    doc_embs = embed_texts(docs)
    q_emb = embed_texts([query])[0]
    sem_sims = doc_embs @ q_emb

    if q_emo_vec is None or np.linalg.norm(q_emo_vec) < 1e-6:
        ranked = np.argsort(sem_sims)[::-1][:top_k]
        return ranked.tolist(), sem_sims[ranked].tolist(), sem_sims[ranked].tolist(), None

    q = q_emo_vec / (np.linalg.norm(q_emo_vec) + 1e-12)
    D = doc_emo_vecs / (np.linalg.norm(doc_emo_vecs, axis=1, keepdims=True) + 1e-12)

    emo_sims = D @ q
    final = alpha * sem_sims + (1 - alpha) * emo_sims
    ranked = np.argsort(final)[::-1][:top_k]

    return ranked.tolist(), final[ranked].tolist(), sem_sims[ranked].tolist(), emo_sims[ranked].tolist()