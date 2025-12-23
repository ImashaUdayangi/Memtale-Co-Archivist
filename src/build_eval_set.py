# src/build_eval_set.py
import pandas as pd
import random
from pathlib import Path

RAW_PATH = Path("dataset/raw/empathetic_stories.csv")
OUT_Q = Path("dataset/eval/queries.csv")
OUT_QRELS = Path("dataset/eval/qrels.csv")

random.seed(42)

def short_semantic_query(situation: str, max_words: int = 10) -> str:
    words = [w for w in situation.split() if w.strip()]
    return " ".join(words[:max_words]).strip()

def main(n_stories: int = 50, extra_same_emotion: int = 3):
    df = pd.read_csv(RAW_PATH)

    # sample a small evaluation set (change n_stories to 30–50 as you want)
    n_stories = min(n_stories, len(df))
    sample = df.sample(n=n_stories, random_state=42).reset_index(drop=True)

    queries = []
    qrels = []

    qid_counter = 1

    # pre-group for same-emotion graded relevance
    by_emotion = {}
    for _, r in df.iterrows():
        by_emotion.setdefault(str(r["emotion_label"]), []).append(int(r["story_id"]))

    for _, row in sample.iterrows():
        sid = int(row["story_id"])
        emo = str(row["emotion_label"])
        sit = str(row["situation"])

        # 1) semantic query
        qid_sem = f"q{qid_counter}"
        qid_counter += 1
        qtext_sem = short_semantic_query(sit)
        if not qtext_sem:
            qtext_sem = f"a personal situation about {emo}"

        queries.append({"query_id": qid_sem, "query_text": qtext_sem, "query_type": "semantic"})

        # exact match relevant
        qrels.append({"query_id": qid_sem, "story_id": sid, "relevance": 2})

        # add a few same-emotion docs as relevance=1
        same_emo_pool = [x for x in by_emotion.get(emo, []) if x != sid]
        random.shuffle(same_emo_pool)
        for rel_sid in same_emo_pool[:extra_same_emotion]:
            qrels.append({"query_id": qid_sem, "story_id": int(rel_sid), "relevance": 1})

        # 2) emotion query
        qid_emo = f"q{qid_counter}"
        qid_counter += 1
        qtext_emo = f"a {emo} experience"

        queries.append({"query_id": qid_emo, "query_text": qtext_emo, "query_type": "emotion"})
        qrels.append({"query_id": qid_emo, "story_id": sid, "relevance": 2})

        same_emo_pool = [x for x in by_emotion.get(emo, []) if x != sid]
        random.shuffle(same_emo_pool)
        for rel_sid in same_emo_pool[:extra_same_emotion]:
            qrels.append({"query_id": qid_emo, "story_id": int(rel_sid), "relevance": 1})

    OUT_Q.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(queries).to_csv(OUT_Q, index=False)
    pd.DataFrame(qrels).to_csv(OUT_QRELS, index=False)

    print("Saved:", OUT_Q, "rows:", len(queries))
    print("Saved:", OUT_QRELS, "rows:", len(qrels))

if __name__ == "__main__":
    main(n_stories=50, extra_same_emotion=3)