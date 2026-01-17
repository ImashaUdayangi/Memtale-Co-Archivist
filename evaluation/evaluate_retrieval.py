import pandas as pd
import numpy as np
import argparse #According to multiple alpha values change
from pathlib import Path 
from src.retrieval import rrf_fusion  # NEW (for fast B4 without recomputing embeddings)
#Remove 16th Jan - from src.retrieval import bm25_sbert_rrf_retrieval #add after Implement rrf_fusion() for B4


from src.summarisation import naive_summary, emotion_aware_summary
from src.setl2 import setl2_emotion_vector
from src.retrieval import (
    build_bm25_index,                   # NEW 16th Jan change
    bm25_rank,                          # NEW 16th Jan change
    emotion_aware_retrieval_cached,     # NEW 16th Jan change
    embed_texts,                        # NEW 16th Jan change
    semantic_retrieval_cached,          # NEW 16th Jan change
) #semantic_retrieval,bm25_retrieval, emotion_aware_retrieval,- Remove 16th Jan change
#emotion_aware_retrieval_vec,build_label_space, onehot - removed as no more onehot - 15th Jan


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
    parser.add_argument("--alpha", type=float, default=0.9, help="Weight for semantic vs emotion (0-1)")
    parser.add_argument("--index", type=str, default="full",
                    choices=["full", "aware", "sit_aware"],
                    help="Indexing strategy for baselines (B1/B2/B4)") #add after Implement rrf_fusion() for B4 , Indexing strategy experiment
    args = parser.parse_args()
    alpha = args.alpha
    index_mode = args.index 

    CANDIDATE_K = 200  # 17th Jan - only rerank top-200 candidates using emotion fusion

    stories = pd.read_csv("dataset/raw/empathetic_stories.csv")  
    queries = pd.read_csv("dataset/eval/queries.csv")
    qrels = pd.read_csv("dataset/eval/qrels.csv")

    # Build text collections
    story_ids = stories["story_id"].astype(int).tolist()
    #16th Jan Remove - full_texts = (stories["situation"].fillna("") + " " + stories["Story"].fillna("")).tolist()

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

    naive_summaries = summaries["naive_summary"].tolist()
    aware_summaries = summaries["aware_summary"].tolist()

    index_texts = build_index_texts(index_mode, stories, summaries) #add after Implement rrf_fusion() for B4 , Indexing strategy experiment

    bm25_index = build_bm25_index(index_texts)  # NEW: build once - 16th Jan

    #16th Jan change to fast retrieval with precomputed embeddings
    # ---------- NEW: Precompute embeddings once (FAST) ----------
    index_doc_embs = embed_texts(index_texts)          
    naive_doc_embs = embed_texts(naive_summaries)      
    aware_doc_embs = embed_texts(aware_summaries)      
    # ----------------------------------------------------------

    aware_emotions = [setl2_emotion_vector(s) for s in aware_summaries]

    # Precompute qrels map
    qrels_map = {}
    for qid, grp in qrels.groupby("query_id"):
        rel_dict = {int(r.story_id): int(r.relevance) for _, r in grp.iterrows()}
        qrels_map[qid] = rel_dict

    results = []

    #16th Jan change to fast retrieval with precomputed embeddings
    query_texts = queries["query_text"].astype(str).tolist()
    query_embs = embed_texts(query_texts)  # (200, d)

    for i, row in enumerate(queries.itertuples(index=False)):
        qid = row.query_id
        qtext = row.query_text
        qtype = row.query_type
    #^ Replace 16th Jan -for _, row in queries.iterrows():
    #     qid = row["query_id"]
    #     qtext = row["query_text"]
    #     qtype = row["query_type"]

        q_emb = query_embs[i]
        #^ Replace 16th jan - q_emb = embed_texts([qtext])[0]  # NEW: compute once per query - 16th Jan

        #rel_dict = qrels_map[qid] // 15th Jan changed as If ever a query is missing, it will crash.
        rel_dict = qrels_map.get(qid, {})
        if not rel_dict:
            continue
        relevant_set = set(rel_dict.keys())

        # B1 BM25 {index_mode}
        #16th Jan Replace - idx, _ = bm25_retrieval(qtext, index_texts, top_k=len(index_texts)) #index_texts - add after Implement rrf_fusion() for B4 , Indexing strategy experiment
        bm25_scores = bm25_rank(bm25_index, qtext)
        idx = np.argsort(bm25_scores)[::-1].tolist()
        ranked_ids = [story_ids[i] for i in idx]
        results.append((f"B1_BM25_{index_mode}", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        # B2 semantic {index_mode}
        #16th Jan Replace - idx, _ = semantic_retrieval(qtext, index_texts, top_k=len(index_texts)) #index_texts - add after Implement rrf_fusion() for B4 , Indexing strategy experiment
        idx, _ = semantic_retrieval_cached(q_emb, index_doc_embs, top_k=len(index_texts))  # NEW 16th Jan
        ranked_ids = [story_ids[i] for i in idx]
        results.append((f"B2_Sem_{index_mode}", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        #add after Implement rrf_fusion() for B4
        # B4 fusion BM25 + SBERT (RRF) on current index texts
        #16th Jan Replace - idx, _ = bm25_sbert_rrf_retrieval(qtext, index_texts, top_k=len(index_texts))
        bm25_scores = bm25_rank(bm25_index, qtext)
        bm25_ranked = np.argsort(bm25_scores)[::-1].tolist()

        sem_sims = index_doc_embs @ q_emb
        sem_ranked = np.argsort(sem_sims)[::-1].tolist()

        fused_scores = np.array(rrf_fusion([bm25_ranked, sem_ranked], k=60))
        idx = np.argsort(fused_scores)[::-1].tolist()
        #----

        ranked_ids = [story_ids[i] for i in idx]
        results.append((f"B4_RRF_{index_mode}", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        # B3 semantic naive summaries
        #16th Jan Replace - idx, _ = semantic_retrieval(qtext, naive_summaries, top_k=len(naive_summaries))
        idx, _ = semantic_retrieval_cached(q_emb, naive_doc_embs, top_k=len(naive_summaries))  # NEW 16th Jan
        ranked_ids = [story_ids[i] for i in idx]
        results.append(("B3_Sem_naiveSum", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        # 17th Jan Change Whole Our block
        # A) If query_type == semantic -> semantic only (alpha=1.0)
        # B) Two-stage: rerank only top CANDIDATE_K semantic candidates with emotion fusion

        # Stage-1: semantic ranking on current index texts (FAST)
        sem_sims_all = index_doc_embs @ q_emb
        sem_ranked_all = np.argsort(sem_sims_all)[::-1].tolist()

        # Candidate set (top-K from semantic)
        cand = sem_ranked_all[:min(CANDIDATE_K, len(sem_ranked_all))]

        if qtype == "semantic":
            # A) semantic queries: do NOT inject emotion (prevents emotion noise)
            idx = sem_ranked_all
        else:
            # emotion_intent queries: apply fusion only within candidate set
            # q_emotion = setl2_emotion_vector(qtext) - Replace 17th Jan 
            # Emotion-aware query expansion (text-only, no labels)
            qtext_for_emotion = (
                f"{qtext}. This experience involves emotions such as "
                "sadness, joy, anger, fear, trust, disgust, surprise, and anticipation."
            )
            q_emotion = setl2_emotion_vector(qtext_for_emotion)

            # build candidate-only arrays
            cand_doc_embs = index_doc_embs[cand]
            cand_doc_emotions = [aware_emotions[j] for j in cand]

            # fusion only within candidates (top_k = len(cand))
            cand_ranked, _, _, _ = emotion_aware_retrieval_cached(
                q_emb=q_emb,
                doc_embs=cand_doc_embs,
                doc_emotions=cand_doc_emotions,
                query_emotion=q_emotion,
                alpha=alpha,
                top_k=len(cand)
            )

            # cand_ranked contains indices 0..len(cand)-1, map back to original doc ids
            reranked_cand = [cand[j] for j in cand_ranked]

            # final list: reranked candidates first + the rest (keeps full ranking length)
            rest = [x for x in sem_ranked_all if x not in set(cand)]
            idx = reranked_cand + rest

        ranked_ids = [story_ids[i] for i in idx]
        results.append((f"Ours_Sem+Emo_{index_mode}", qid, qtype,
                        precision_at_k(ranked_ids, relevant_set, 5),
                        ndcg_at_k(ranked_ids, rel_dict, 5),
                        mrr(ranked_ids, relevant_set)))

        # ^^^ Added above # Ours semantic+emotion aware summaries retrieval (NO LABEL LEAKAGE)
        # # Always derive emotion from the query text (never from gold labels / one-hot)
        # # Ours: semantic from index_doc_embs (depends on index_mode) + emotion from aware_emotions
        # q_emotion = setl2_emotion_vector(qtext)

        # idx, _, _, _ = emotion_aware_retrieval_cached(
        #     q_emb=q_emb,
        #     doc_embs=index_doc_embs,       #<-- 17th Jan CHANGE:use index embeddings not - aware_doc_embs,
        #     doc_emotions=aware_emotions,
        #     query_emotion=q_emotion,
        #     alpha=alpha,
        #     top_k=len(index_texts)
        # )

        # #16th Jan Replace with cached version above
        # # idx, _, _, _ = emotion_aware_retrieval(
        # #     qtext,
        # #     aware_summaries,
        # #     aware_emotions,
        # #     q_emotion,
        # #     alpha=alpha,
        # #     top_k=len(aware_summaries)
        # # )

        # #15th Jan REMOVE 
        # # if qtype == "emotion":
        # #     # parse label from: "a <emotion> experience"
        # #     emo = str(qtext).lower().replace("a ", "").replace(" experience", "").strip()
        # #     q_vec = onehot(emo, label2i)

        # #     idx, _, _, _ = emotion_aware_retrieval_vec(
        # #         qtext,
        # #         aware_summaries,
        # #         doc_label_vecs,
        # #         q_vec,
        # #         alpha=alpha, #According to multiple alpha values change
        # #         top_k=len(aware_summaries)
        # #     )
        # # else:
        # #     # semantic queries: keep your original NRC8-based emotion aware retrieval (or semantic-only)
        # #     q_emotion = setl2_emotion_vector(qtext)
        # #     idx, _, _, _ = emotion_aware_retrieval(
        # #         qtext,
        # #         aware_summaries,
        # #         aware_emotions,
        # #         q_emotion,
        # #         alpha=alpha, #According to multiple alpha values change
        # #         top_k=len(aware_summaries)
        # #     )
        # ranked_ids = [story_ids[i] for i in idx]
        # results.append((f"Ours_Sem+Emo_{index_mode}", qid, qtype,
        #                 precision_at_k(ranked_ids, relevant_set, 5),
        #                 ndcg_at_k(ranked_ids, rel_dict, 5),
        #                 mrr(ranked_ids, relevant_set)))

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