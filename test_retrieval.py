import pandas as pd
from src.summarisation import naive_summary, emotion_aware_summary
from src.setl2 import setl2_emotion_vector
from src.retrieval import tfidf_retrieval, semantic_retrieval, emotion_aware_retrieval

df = pd.read_csv("dataset/raw_stories.csv")

# Prepare texts
full_texts = df["text"].tolist()

# Summaries
naive_summaries = [naive_summary(t) for t in full_texts]
aware_summaries = [emotion_aware_summary(t) for t in full_texts]

# Emotion vectors on summaries (this is what we retrieve over)
aware_emotions = [setl2_emotion_vector(s) for s in aware_summaries]

# Example queries (one semantic, one emotion-focused)
queries = [
    "moving to a new city and leaving family",
    "a sad memory about losing someone"
]

for q in queries:
    print("\n========================")
    print("QUERY:", q)

    # B1: keyword TF-IDF on full text
    idx, scores = tfidf_retrieval(q, full_texts, top_k=3)
    print("\nB1 TF-IDF (full text):")
    for i, s in zip(idx, scores):
        print(f"  -> {df.loc[i,'title']}  score={s:.3f}")

    # B2: semantic on full text
    idx, scores = semantic_retrieval(q, full_texts, top_k=3)
    print("\nB2 Semantic (full text):")
    for i, s in zip(idx, scores):
        print(f"  -> {df.loc[i,'title']}  score={s:.3f}")

    # B3: semantic on naive summaries
    idx, scores = semantic_retrieval(q, naive_summaries, top_k=3)
    print("\nB3 Semantic (naive summaries):")
    for i, s in zip(idx, scores):
        print(f"  -> {df.loc[i,'title']}  score={s:.3f}")

    # Ours: semantic + emotion on aware summaries
    q_emotion = setl2_emotion_vector(q)
    idx, final, sem, emo = emotion_aware_retrieval(
        q, aware_summaries, aware_emotions, q_emotion, alpha=0.7, top_k=3
    )
    print("\nOURS Semantic+Emotion (aware summaries):")
    for i, fs, ss, es in zip(idx, final, sem, emo):
        print(f"  -> {df.loc[i,'title']} final={fs:.3f} sem={ss:.3f} emo={es:.3f}")