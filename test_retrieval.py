# test_retrieval.py
import pandas as pd

from src.summarisation import naive_summary, emotion_aware_summary
from src.setl2 import setl2_emotion_vector
from src.retrieval import (
    bm25_retrieval,
    semantic_retrieval,
    emotion_aware_retrieval,
    bm25_sbert_rrf_retrieval
)

def main():
    df = pd.read_csv("dataset/raw/raw_stories.csv")

    # Expect columns: title, text
    if "text" not in df.columns:
        raise ValueError("dataset/raw/raw_stories.csv must contain a 'text' column")
    if "title" not in df.columns:
        df["title"] = [f"Story {i}" for i in range(len(df))]

    # Prepare full texts
    full_texts = df["text"].fillna("").astype(str).tolist()

    # Generate summaries
    print("Generating naive summaries...")
    naive_summaries = [naive_summary(t) for t in full_texts]

    print("Generating emotion-aware summaries...")
    aware_summaries = [emotion_aware_summary(t, base_summary=ns) for t, ns in zip(full_texts, naive_summaries)]

    # Emotion vectors from emotion-aware summaries (doc-side emotion)
    aware_emotions = [setl2_emotion_vector(s) for s in aware_summaries]

    # Example queries (semantic + emotion-driven)
    queries = [
        "moving to a new city and leaving family",
        "a sad memory about losing someone",
        "a proud moment after hard work",
        "something exciting that happened unexpectedly"
    ]

    for q in queries:
        print("\n" + "=" * 60)
        print("QUERY:", q)

        # B1: BM25 on full text
        idx, scores = bm25_retrieval(q, full_texts, top_k=3)
        print("\nB1 BM25 (full text):")
        for i, s in zip(idx, scores):
            print(f"  -> {df.loc[i,'title']}  score={s:.4f}")

        # B2: Semantic on full text
        idx, scores = semantic_retrieval(q, full_texts, top_k=3)
        print("\nB2 Semantic (full text):")
        for i, s in zip(idx, scores):
            print(f"  -> {df.loc[i,'title']}  score={s:.4f}")

        # B3: Semantic on naive summaries
        idx, scores = semantic_retrieval(q, naive_summaries, top_k=3)
        print("\nB3 Semantic (naive summaries):")
        for i, s in zip(idx, scores):
            print(f"  -> {df.loc[i,'title']}  score={s:.4f}")

        # B4: BM25 + SBERT fusion (RRF) on full text
        idx, scores = bm25_sbert_rrf_retrieval(q, full_texts, top_k=3, rrf_k=60)
        print("\nB4 BM25 + SBERT (RRF fusion, full text):")
        for i, s in zip(idx, scores):
            print(f"  -> {df.loc[i,'title']}  fused={s:.6f}")

        # Ours: Semantic + Emotion on emotion-aware summaries
        q_emotion = setl2_emotion_vector(q)
        idx, final, sem, emo = emotion_aware_retrieval(
            query=q,
            docs=aware_summaries,
            doc_emotions=aware_emotions,
            query_emotion=q_emotion,
            alpha=0.7,
            top_k=3
        )

        print("\nOURS Semantic + Emotion (emotion-aware summaries):")
        # emo can be None if query has no emotion signal (fallback mode)
        if emo is None:
            for i, fs, ss in zip(idx, final, sem):
                print(f"  -> {df.loc[i,'title']}  final={fs:.4f}  sem={ss:.4f}  emo=None")
        else:
            for i, fs, ss, es in zip(idx, final, sem, emo):
                print(f"  -> {df.loc[i,'title']}  final={fs:.4f}  sem={ss:.4f}  emo={es:.4f}")

if __name__ == "__main__":
    main()
