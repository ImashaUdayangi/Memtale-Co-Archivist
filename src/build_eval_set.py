# src/build_eval_set.py
import pandas as pd
import random
from pathlib import Path

RAW_PATH = Path("dataset/raw/empathetic_stories.csv")
OUT_Q = Path("dataset/eval/queries.csv")
OUT_QRELS = Path("dataset/eval/qrels.csv")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def make_semantic_query(situation: str, max_words: int = 12) -> str:
    """
    Simple semantic query derived from the situation text.
    Keeps it short and retrieval-like.
    """
    if not isinstance(situation, str):
        return ""
    text = situation.replace("\n", " ").strip()
    words = [w for w in text.split() if w]
    return " ".join(words[:max_words]).strip()


def main(
    n_stories: int = 50,
    extra_same_emotion: int = 3,
    max_emotion_pool_scan: int | None = None,
):
    """
    n_stories: how many stories to sample for evaluation set (e.g., 50 or 100)
    extra_same_emotion: how many additional same-emotion docs to label as relevance=1
    max_emotion_pool_scan: optionally limit how many rows we scan to build emotion pools (speed)
    """
    df = pd.read_csv(RAW_PATH)

    required_cols = {"story_id", "emotion_label", "situation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {RAW_PATH}: {missing}")

    # Ensure correct dtypes
    df["story_id"] = df["story_id"].astype(int)
    df["emotion_label"] = df["emotion_label"].astype(str)
    df["situation"] = df["situation"].astype(str)

    # Build an emotion->story_id pool for graded relevance (relevance=1)
    # For large datasets, you can optionally scan only the first N rows to build pools
    pool_df = df if max_emotion_pool_scan is None else df.head(max_emotion_pool_scan)

    by_emotion: dict[str, list[int]] = {}
    for _, r in pool_df.iterrows():
        emo = str(r["emotion_label"]).strip()
        sid = int(r["story_id"])
        by_emotion.setdefault(emo, []).append(sid)

    # Sample evaluation stories
    n_stories = min(n_stories, len(df))
    sample = df.sample(n=n_stories, random_state=RANDOM_SEED).reset_index(drop=True)

    queries = []
    qrels = []
    qid_counter = 1

    for _, row in sample.iterrows():
        sid = int(row["story_id"])
        emo = str(row["emotion_label"]).strip()
        sit = str(row["situation"]).strip()

        # -------------------
        # 1) Semantic query
        # -------------------
        qid_sem = f"q{qid_counter}"
        qid_counter += 1
        qtext_sem = make_semantic_query(sit)

        # fallback if empty
        if not qtext_sem:
            qtext_sem = f"a personal situation about {emo}"

        queries.append(
            {"query_id": qid_sem, "query_text": qtext_sem, "query_type": "semantic"}
        )

        # qrels: exact story is relevance 2
        qrels.append({"query_id": qid_sem, "story_id": sid, "relevance": 2})

        # add a few same-emotion docs as relevance 1
        same_pool = [x for x in by_emotion.get(emo, []) if x != sid]
        random.shuffle(same_pool)
        for rel_sid in same_pool[:extra_same_emotion]:
            qrels.append({"query_id": qid_sem, "story_id": int(rel_sid), "relevance": 1})

        # -------------------
        # 2) Emotion query
        # -------------------
        qid_emo = f"q{qid_counter}"
        qid_counter += 1
        qtext_emo = f"a {emo} experience"

        queries.append(
            {"query_id": qid_emo, "query_text": qtext_emo, "query_type": "emotion"}
        )

        qrels.append({"query_id": qid_emo, "story_id": sid, "relevance": 2})

        same_pool = [x for x in by_emotion.get(emo, []) if x != sid]
        random.shuffle(same_pool)
        for rel_sid in same_pool[:extra_same_emotion]:
            qrels.append({"query_id": qid_emo, "story_id": int(rel_sid), "relevance": 1})

    # Save
    OUT_Q.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(queries).to_csv(OUT_Q, index=False)
    pd.DataFrame(qrels).to_csv(OUT_QRELS, index=False)

    print(f"Saved: {OUT_Q} rows: {len(queries)}")
    print(f"Saved: {OUT_QRELS} rows: {len(qrels)}")
    print(f"Eval stories sampled: {n_stories} (2 queries each -> {2*n_stories})")
    print(f"extra_same_emotion per query: {extra_same_emotion}")


if __name__ == "__main__":
    # Good defaults for first run
    main(n_stories=50, extra_same_emotion=3)