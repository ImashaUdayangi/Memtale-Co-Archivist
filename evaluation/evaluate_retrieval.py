import pandas as pd
import numpy as np
import argparse #According to multiple alpha values change
from pathlib import Path 
from src.retrieval import bm25_sbert_rrf_retrieval #add after Implement rrf_fusion() for B4

from src.summarisation import naive_summary, emotion_aware_summary
from src.setl2 import setl2_emotion_vector
from src.retrieval import (
    bm25_retrieval,
    semantic_retrieval,
    emotion_aware_retrieval,
) #emotion_aware_retrieval_vec,build_label_space, onehot - removed as no more onehot - 15th Jan


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

#add after Implement rrf_fusion() for B4 , a helper function - Indexing strategy experiment 
def build_index_texts(index_mode: str, stories: pd.DataFrame, summaries: pd.DataFrame):
    """
    index_mode:
      - full: situation + Story
      - aware: aware_summary only
      - sit_aware: situation + aware_summary
    """
    story_ids = stories["story_id"].astype(int).tolist()

    if index_mode == "full":
        return (stories["situation"].fillna("") + " " + stories["Story"].fillna("")).tolist()

    # align summaries to the same story_id order
    summaries = summaries.set_index("story_id").loc[story_ids].reset_index()

    if index_mode == "aware":
        return summaries["aware_summary"].fillna("").tolist()

    if index_mode == "sit_aware":
        sit = stories["situation"].fillna("").tolist()
        aware = summaries["aware_summary"].fillna("").tolist()
        return [f"{a} {b}".strip() for a, b in zip(sit, aware)]

    raise ValueError(f"Unknown index_mode: {index_mode}")

def main():
    #According to multiple alpha values change
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.8, help="Weight for semantic vs emotion (0-1)")
    parser.add_argument("--index", type=str, default="full",
                    choices=["full", "aware", "sit_aware"],
                    help="Indexing strategy for baselines (B1/B2/B4)") #add after Implement rrf_fusion() for B4 , Indexing strategy experiment
    args = parser.parse_args()
    alpha = args.alpha
    index_mode = args.index 

    stories = pd.read_csv("dataset/raw/empathetic_stories.csv")  
    queries = pd.read_csv("dataset/eval/queries.csv")
    qrels = pd.read_csv("dataset/eval/qrels.csv")

    # Build text collections
    story_ids = stories["story_id"].astype(int).tolist()
    full_texts = (stories["situation"].fillna("") + " " + stories["Story"].fillna("")).tolist()

    #15th Jan REMOVE as will never use gold labels for retrieval
    #labels, label2i = build_label_space(stories["emotion_label"].tolist())
    #doc_label_vecs = np.vstack([onehot(lab, label2i) for lab in stories["emotion_label"].tolist()])

    #REMOVE this (current, slow)
    #//naive_summaries = [naive_summary(t) for t in full_texts]
    #//aware_summaries = [emotion_aware_summary(t) for t in full_texts]

    #Replace with(fast,cached) 48-55
    summaries = pd.read_csv("dataset/cache/summaries.csv")

    # IMPORTANT: keep the same order as stories
    summaries = summaries.set_index("story_id").loc[story_ids].reset_index()

    index_texts = build_index_texts(index_mode, stories, summaries) #add after Implement rrf_fusion() for B4 , Indexing strategy experiment

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
        idx, _ = bm25_retrieval(qtext, index_texts, top_k=len(index_texts)) #index_texts - add after Implement rrf_fusion() for B4 , Indexing strategy experiment
        ranked_ids = [story_ids[i] for i in idx]
        results.append(("B1_BM25_full", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        # B2 semantic full
        idx, _ = semantic_retrieval(qtext, index_texts, top_k=len(index_texts)) #index_texts - add after Implement rrf_fusion() for B4 , Indexing strategy experiment
        ranked_ids = [story_ids[i] for i in idx]
        results.append(("B2_Sem_full", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        #add after Implement rrf_fusion() for B4
        # B4 fusion BM25 + SBERT (RRF) on current index texts
        idx, _ = bm25_sbert_rrf_retrieval(qtext, index_texts, top_k=len(index_texts))
        ranked_ids = [story_ids[i] for i in idx]
        results.append(("B4_RRF_BM25+SBERT", qid, qtype,
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

        # Ours semantic+emotion aware summaries retrieval (NO LABEL LEAKAGE)
        # Always derive emotion from the query text (never from gold labels / one-hot)
        q_emotion = setl2_emotion_vector(qtext)

        idx, _, _, _ = emotion_aware_retrieval(
            qtext,
            aware_summaries,
            aware_emotions,
            q_emotion,
            alpha=alpha,
            top_k=len(aware_summaries)
        )

        #15th Jan REMOVE 
        # if qtype == "emotion":
        #     # parse label from: "a <emotion> experience"
        #     emo = str(qtext).lower().replace("a ", "").replace(" experience", "").strip()
        #     q_vec = onehot(emo, label2i)

        #     idx, _, _, _ = emotion_aware_retrieval_vec(
        #         qtext,
        #         aware_summaries,
        #         doc_label_vecs,
        #         q_vec,
        #         alpha=alpha, #According to multiple alpha values change
        #         top_k=len(aware_summaries)
        #     )
        # else:
        #     # semantic queries: keep your original NRC8-based emotion aware retrieval (or semantic-only)
        #     q_emotion = setl2_emotion_vector(qtext)
        #     idx, _, _, _ = emotion_aware_retrieval(
        #         qtext,
        #         aware_summaries,
        #         aware_emotions,
        #         q_emotion,
        #         alpha=alpha, #According to multiple alpha values change
        #         top_k=len(aware_summaries)
        #     )
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

    #add after Implement rrf_fusion() for B4 , Indexing strategy experiment
    # Save to CSV for your report tables/plots
    run_dir = Path(f"evaluation/runs/index_{index_mode}/alpha_{alpha}")
    run_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(run_dir / "per_query_results.csv", index=False)
    summary_all.to_csv(run_dir / "summary_overall.csv", index=False)
    summary_by_type.to_csv(run_dir / "summary_by_type.csv", index=False)


if __name__ == "__main__":
    main()