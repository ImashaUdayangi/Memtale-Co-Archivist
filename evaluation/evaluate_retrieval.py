import pandas as pd
import numpy as np

from src.summarisation import naive_summary, emotion_aware_summary
from src.setl2 import setl2_emotion_vector
from src.retrieval import (
    bm25_retrieval,
    semantic_retrieval,
    emotion_aware_retrieval,
    emotion_aware_retrieval_vec,
    build_label_space,
    onehot
)


def precision_at_k(ranked_ids, relevant_set, k=5):
    topk = ranked_ids[:k]
    hit = sum(1 for sid in topk if sid in relevant_set)
    return hit / k


def mrr(ranked_ids, relevant_set):
    for i, sid in enumerate(ranked_ids, start=1):
        if sid in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids, rel_dict, k=5):
    def dcg(items):
        score = 0.0
        for i, sid in enumerate(items[:k], start=1):
            rel = rel_dict.get(sid, 0)
            score += (2**rel - 1) / np.log2(i + 1)
        return score

    ideal = sorted(rel_dict.keys(), key=lambda x: rel_dict[x], reverse=True)
    denom = dcg(ideal)
    return (dcg(ranked_ids) / denom) if denom > 0 else 0.0


def main():
    stories = pd.read_csv("dataset/raw/empathetic_stories.csv")  
    queries = pd.read_csv("dataset/eval/queries.csv")
    qrels = pd.read_csv("dataset/eval/qrels.csv")

    # Build text collections
    story_ids = stories["story_id"].astype(int).tolist()
    full_texts = (stories["situation"].fillna("") + " " + stories["Story"].fillna("")).tolist()

    labels, label2i = build_label_space(stories["emotion_label"].tolist())
    doc_label_vecs = np.vstack([onehot(lab, label2i) for lab in stories["emotion_label"].tolist()])

    #REMOVE this (current, slow)
    #//naive_summaries = [naive_summary(t) for t in full_texts]
    #//aware_summaries = [emotion_aware_summary(t) for t in full_texts]

    #Replace with(fast,cached) 48-55
    summaries = pd.read_csv("dataset/cache/summaries.csv")

    # IMPORTANT: keep the same order as stories
    summaries = summaries.set_index("story_id").loc[story_ids].reset_index()

    naive_summaries = summaries["naive_summary"].tolist()
    aware_summaries = summaries["aware_summary"].tolist()
    aware_emotions = [setl2_emotion_vector(s) for s in aware_summaries]

    # Precompute qrels map
    qrels_map = {}
    for qid, grp in qrels.groupby("query_id"):
        rel_dict = {int(r.story_id): int(r.relevance) for _, r in grp.iterrows()}
        qrels_map[qid] = rel_dict

    results = []

    for _, row in queries.iterrows():
        qid = row["query_id"]
        qtext = row["query_text"]
        qtype = row["query_type"]

        rel_dict = qrels_map[qid]
        relevant_set = set(rel_dict.keys())

        # B1 BM25 full
        idx, _ = bm25_retrieval(qtext, full_texts, top_k=len(full_texts))
        ranked_ids = [story_ids[i] for i in idx]
        results.append(("B1_BM25_full", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        # B2 semantic full
        idx, _ = semantic_retrieval(qtext, full_texts, top_k=len(full_texts))
        ranked_ids = [story_ids[i] for i in idx]
        results.append(("B2_Sem_full", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        # B3 semantic naive summaries
        idx, _ = semantic_retrieval(qtext, naive_summaries, top_k=len(naive_summaries))
        ranked_ids = [story_ids[i] for i in idx]
        results.append(("B3_Sem_naiveSum", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        # Ours semantic+emotion aware summaries
        if qtype == "emotion":
            # parse label from: "a <emotion> experience"
            emo = str(qtext).lower().replace("a ", "").replace(" experience", "").strip()
            q_vec = onehot(emo, label2i)

            idx, _, _, _ = emotion_aware_retrieval_vec(
                qtext,
                aware_summaries,
                doc_label_vecs,
                q_vec,
                alpha=0.7,
                top_k=len(aware_summaries)
            )
        else:
            # semantic queries: keep your original NRC8-based emotion aware retrieval (or semantic-only)
            q_emotion = setl2_emotion_vector(qtext)
            idx, _, _, _ = emotion_aware_retrieval(
                qtext,
                aware_summaries,
                aware_emotions,
                q_emotion,
                alpha=0.7,
                top_k=len(aware_summaries)
            )
        ranked_ids = [story_ids[i] for i in idx]
        results.append(("Ours_Sem+Emo_awareSum", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

    df = pd.DataFrame(results, columns=["system", "query_id", "query_type", "P@5", "nDCG@5", "MRR"])

    # Aggregate
    summary_all = df.groupby("system")[["P@5", "nDCG@5", "MRR"]].mean().reset_index()
    summary_by_type = df.groupby(["system", "query_type"])[["P@5", "nDCG@5", "MRR"]].mean().reset_index()

    print("\n=== Overall Results ===")
    print(summary_all.to_string(index=False))

    print("\n=== Results by Query Type ===")
    print(summary_by_type.to_string(index=False))

    # Save to CSV for your report tables/plots
    df.to_csv("evaluation/per_query_results.csv", index=False)
    summary_all.to_csv("evaluation/summary_overall.csv", index=False)
    summary_by_type.to_csv("evaluation/summary_by_type.csv", index=False)


if __name__ == "__main__":
    main()