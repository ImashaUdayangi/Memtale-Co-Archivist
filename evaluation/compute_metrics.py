# evaluation/compute_metrics.py

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

def dcg(rels):
    return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k_from_qrels(qid, ranked_docs, qrels_dict, k=5):
    # gains for retrieved docs (missing judged docs -> 0)
    retrieved_rels = [qrels_dict[qid].get(doc_id, 0) for doc_id in ranked_docs[:k]]

    # ideal gains from ALL judged docs for that query
    all_judged_rels = list(qrels_dict[qid].values())
    ideal_rels = sorted(all_judged_rels, reverse=True)[:k]

    if sum(ideal_rels) == 0:
        return 0.0

    return dcg(retrieved_rels) / dcg(ideal_rels)

def mrr_from_qrels(qid, ranked_docs, qrels_dict):
    for i, doc_id in enumerate(ranked_docs):
        if qrels_dict[qid].get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


QRELS_PATH = Path("dataset/eval_v3/qrels.csv")
RUNS_DIR = Path("evaluation/results")

qrels = pd.read_csv(QRELS_PATH)

# build qrels dict
qrels_dict = defaultdict(dict)
for _, row in qrels.iterrows():
    qrels_dict[row["query_id"]][int(row["story_id"])] = int(row["relevance"])

run_files = {
    "BM25": "bm25_run.csv",
    "SBERT": "sbert_run.csv",
    "RRF": "rrf_run.csv",
    "OURS": "ours_run.csv",
}

results = []

for method, fname in run_files.items():
    run = pd.read_csv(RUNS_DIR / fname)

    ndcgs = []
    mrrs = []

    for qid, group in run.groupby("query_id"):
        ranked_docs = group.sort_values("rank")["story_id"].astype(int).tolist()

        ndcgs.append(ndcg_at_k_from_qrels(qid, ranked_docs, qrels_dict, k=5))
        mrrs.append(mrr_from_qrels(qid, ranked_docs, qrels_dict))

    results.append({
        "Method": method,
        "nDCG@5": round(float(np.mean(ndcgs)), 4),
        "MRR": round(float(np.mean(mrrs)), 4),
    })

results_df = pd.DataFrame(results)
OUT_PATH = RUNS_DIR / "evaluation_summary.csv"
results_df.to_csv(OUT_PATH, index=False)

print("\n=== Evaluation Results (Correct nDCG@5) ===")
print(results_df)
print(f"\nSaved to {OUT_PATH}")
