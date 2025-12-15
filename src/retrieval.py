# src/retrieval.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# SBERT model (good default)
_sbert = SentenceTransformer("all-MiniLM-L6-v2")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0][0])

# -------------------------
# B1: Keyword retrieval (TF-IDF)
# -------------------------
def tfidf_retrieval(query: str, docs: list[str], top_k: int = 5):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(docs + [query])
    doc_mat = X[:-1]
    q_vec = X[-1]
    sims = cosine_similarity(q_vec, doc_mat)[0]
    ranked = np.argsort(sims)[::-1][:top_k]
    return ranked.tolist(), sims[ranked].tolist()

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
    q_e = q_e / (np.linalg.norm(q_e) + 1e-12)

    emo_sims = []
    for e in doc_emotions:
        d_e = np.array([e[k] for k in keys], dtype=float)
        d_e = d_e / (np.linalg.norm(d_e) + 1e-12)
        emo_sims.append(float(d_e @ q_e))

    emo_sims = np.array(emo_sims)

    final = alpha * sem_sims + (1 - alpha) * emo_sims
    ranked = np.argsort(final)[::-1][:top_k]
    return ranked.tolist(), final[ranked].tolist(), sem_sims[ranked].tolist(), emo_sims[ranked].tolist()