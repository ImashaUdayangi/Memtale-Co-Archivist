import os
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# ---------------- CONFIG ----------------
K = 100
DATASET_DIR = "dataset/eval_v3"
CORPUS_PATH = "dataset/raw/empathetic_stories.csv"
RESULTS_DIR = "evaluation/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------
# Load queries + qrels (for later metrics scripts)
queries = pd.read_csv(os.path.join(DATASET_DIR, "queries.csv"))

# Load corpus (EmpatheticDialogues stories)
stories = pd.read_csv(CORPUS_PATH)

# Ensure required columns exist
required_cols = {"story_id", "emotion_label", "Story"}
missing = required_cols - set(stories.columns)
if missing:
    raise ValueError(f"Corpus file is missing columns: {missing}. Found: {list(stories.columns)}")

# Normalize column names for this script
stories = stories.rename(columns={"emotion_label": "emotion", "Story": "text"})

# Fill NaNs just in case
stories["text"] = stories["text"].fillna("").astype(str)
stories["emotion"] = stories["emotion"].fillna("").astype(str)

# BM25 tokenization
stories["tokens"] = stories["text"].str.lower().str.split()
bm25 = BM25Okapi(stories["tokens"].tolist())

# SBERT
sbert = SentenceTransformer("all-MiniLM-L6-v2")
story_embeddings = sbert.encode(
    stories["text"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

# ---------------------------------------
def rrf(rank1, rank2, k=60):
    """Reciprocal Rank Fusion over index lists."""
    scores = {}
    for i, doc_idx in enumerate(rank1):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + i + 1)
    for i, doc_idx in enumerate(rank2):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------------------------------------
all_runs = {
    "bm25": [],
    "sbert": [],
    "rrf": [],
    "ours": []
}

for _, q in queries.iterrows():
    qid = q["query_id"]
    qtext = str(q["query_text"])
    qtype = str(q["query_type"])
    target_emotion = str(q["target_emotion"])

    q_tokens = qtext.lower().split()

    # -------- Stage 1: BM25 over full corpus --------
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:K]

    # -------- SBERT over full corpus (baseline + for rerank features) --------
    q_emb = sbert.encode(qtext, convert_to_tensor=True)
    sbert_scores = util.cos_sim(q_emb, story_embeddings)[0].cpu().numpy()
    sbert_top_idx = np.argsort(sbert_scores)[::-1][:K]

    # -------- RRF baseline (BM25 + SBERT) --------
    rrf_rank = rrf(bm25_top_idx, sbert_top_idx)

    # -------- OURS: rerank BM25 topK only --------
    # Adaptive weights (NO fixed alpha)
    if qtype == "emotion_intent":
        w_bm25, w_sem, w_em = 0.3, 0.4, 0.3
    else:
        w_bm25, w_sem, w_em = 0.5, 0.4, 0.1

    rerank_scores = {}
    # Normalize BM25 scores within topK to avoid scale dominance
    bm25_top_vals = bm25_scores[bm25_top_idx]
    bm25_min, bm25_max = float(bm25_top_vals.min()), float(bm25_top_vals.max())
    denom = (bm25_max - bm25_min) if (bm25_max - bm25_min) > 1e-9 else 1.0

    for idx in bm25_top_idx:
        bm25_norm = (float(bm25_scores[idx]) - bm25_min) / denom
        sem = float(sbert_scores[idx])
        em_match = 1.0 if stories.iloc[idx]["emotion"] == target_emotion else 0.0

        rerank_scores[idx] = (w_bm25 * bm25_norm) + (w_sem * sem) + (w_em * em_match)

    reranked = sorted(rerank_scores.items(), key=lambda x: x[1], reverse=True)

    # Save runs as: query_id, story_id, rank
    for rank, idx in enumerate(bm25_top_idx, start=1):
        all_runs["bm25"].append([qid, int(stories.iloc[idx]["story_id"]), rank])

    for rank, idx in enumerate(sbert_top_idx, start=1):
        all_runs["sbert"].append([qid, int(stories.iloc[idx]["story_id"]), rank])

    for rank, (idx, _) in enumerate(rrf_rank[:K], start=1):
        all_runs["rrf"].append([qid, int(stories.iloc[idx]["story_id"]), rank])

    for rank, (idx, _) in enumerate(reranked, start=1):
        all_runs["ours"].append([qid, int(stories.iloc[idx]["story_id"]), rank])

# ---------------------------------------
for name, rows in all_runs.items():
    df = pd.DataFrame(rows, columns=["query_id", "story_id", "rank"])
    out_path = os.path.join(RESULTS_DIR, f"{name}_run.csv")
    df.to_csv(out_path, index=False)

print("✅ Saved run files to evaluation/results/:")
for name in all_runs.keys():
    print(f" - {name}_run.csv")
