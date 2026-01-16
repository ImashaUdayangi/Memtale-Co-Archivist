# src/build_eval_set.py

import pandas as pd
import random
from pathlib import Path

RAW_PATH = Path("dataset/raw/empathetic_stories.csv")
OUT_Q = Path("dataset/eval/queries.csv")
OUT_QRELS = Path("dataset/eval/qrels.csv")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def make_semantic_hint(situation: str, max_words: int = 8) -> str:
    """Short hint phrase derived from situation text (used to make emotion queries more unique)."""
    if not isinstance(situation, str):
        return ""
    text = situation.replace("\n", " ").strip()
    words = [w for w in text.split() if w]
    return " ".join(words[:max_words]).strip()

def make_semantic_query(situation: str, max_words: int = 12) -> str:
    """Short semantic query derived from situation text."""
    return make_semantic_hint(situation, max_words=max_words)

# Natural-language emotion-intent prompts (avoid trivial "a <label> experience")
EMO_INTENT_PROMPTS = {
    "afraid": "a memory about feeling anxious and unsafe, like something bad might happen",
    "angry": "a memory about feeling furious after being treated unfairly or disrespected",
    "annoyed": "a memory about being irritated by something small that kept happening",
    "anticipating": "a memory about waiting for something important and feeling on edge",
    "anxious": "a memory about worrying a lot and not being able to relax",
    "apprehensive": "a memory about being uncertain and nervous about what was coming next",
    "ashamed": "a memory about feeling embarrassed and regretting something you did",
    "caring": "a memory about supporting someone and wanting to help them feel better",
    "confident": "a memory about feeling capable and sure you could handle the situation",
    "content": "a memory about feeling calm, satisfied, and at peace with how things were",
    "devastated": "a memory about feeling completely crushed after a big loss or setback",
    "disappointed": "a memory about feeling let down when something didn’t work out",
    "disgusted": "a memory about feeling strongly uncomfortable with something unpleasant",
    "embarrassed": "a memory about feeling awkward after a mistake in front of others",
    "excited": "a memory about feeling thrilled and full of energy about something new",
    "faithful": "a memory about staying loyal and committed even when things were difficult",
    "furious": "a memory about intense anger and wanting to react strongly",
    "grateful": "a memory about appreciating someone’s help or kindness",
    "guilty": "a memory about regretting something and wishing you could undo it",
    "hopeful": "a memory about believing things could improve despite challenges",
    "impressed": "a memory about admiring someone’s skills or a surprising achievement",
    "jealous": "a memory about wanting what someone else had and feeling bothered by it",
    "joyful": "a memory about feeling happy and uplifted by a positive moment",
    "lonely": "a memory about feeling alone or unsupported, even around other people",
    "nostalgic": "a memory about looking back warmly on the past and missing it",
    "prepared": "a memory about feeling ready and organized for something important",
    "proud": "a memory about feeling accomplished after hard work paid off",
    "sad": "a memory about feeling down and hurt, like you lost something important",
    "sentimental": "a memory about deep feelings tied to a meaningful person or moment",
    "surprised": "a memory about something unexpected that changed how you felt",
    "terrified": "a memory about intense fear, like you were in real danger",
    "trusting": "a memory about relying on someone and believing they would support you",
}

def make_emotion_intent_query(emotion_label: str) -> str:
    """Create a natural language emotion-intent query for a given label."""
    emo = (emotion_label or "").strip().lower()
    return EMO_INTENT_PROMPTS.get(
        emo,
        "a memory about strong emotions and how you handled the situation"
    )

def main(
    n_semantic: int = 100,
    n_emotion_intent: int = 100,
    extra_same_emotion: int = 3,
    max_emotion_pool_scan: int | None = None,
):
    df = pd.read_csv(RAW_PATH)

    required_cols = {"story_id", "emotion_label", "situation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {RAW_PATH}: {missing}")

    df["story_id"] = df["story_id"].astype(int)
    df["emotion_label"] = df["emotion_label"].astype(str).str.strip()
    df["situation"] = df["situation"].astype(str)

    # Build emotion -> story_id pool (used ONLY for evaluation/qrels construction)
    pool_df = df if max_emotion_pool_scan is None else df.head(max_emotion_pool_scan)
    by_emotion: dict[str, list[int]] = {}
    for _, r in pool_df.iterrows():
        emo = str(r["emotion_label"]).strip()
        sid = int(r["story_id"])
        by_emotion.setdefault(emo, []).append(sid)

    # Sample stories for semantic + emotion-intent sets
    n_semantic = min(n_semantic, len(df))
    n_emotion_intent = min(n_emotion_intent, max(0, len(df) - n_semantic))

    sem_sample = df.sample(n=n_semantic, random_state=RANDOM_SEED).reset_index(drop=True)
    remaining = df[~df["story_id"].isin(sem_sample["story_id"])].copy()
    emo_sample = remaining.sample(n=n_emotion_intent, random_state=RANDOM_SEED).reset_index(drop=True)

    queries = []
    qrels = []
    qid_counter = 1

    # 1) Semantic queries
    for _, row in sem_sample.iterrows():
        sid = int(row["story_id"])
        emo = str(row["emotion_label"]).strip()
        sit = str(row["situation"]).strip()

        qid = f"q{qid_counter}"
        qid_counter += 1

        qtext = make_semantic_query(sit)
        if not qtext:
            qtext = f"a personal situation where someone feels {emo}"

        queries.append({"query_id": qid, "query_text": qtext, "query_type": "semantic"})

        # qrels: exact story is relevance 2
        qrels.append({"query_id": qid, "story_id": sid, "relevance": 2})

        # add same-emotion docs as relevance 1 (graded relevance)
        same_pool = [x for x in by_emotion.get(emo, []) if x != sid]
        random.shuffle(same_pool)
        for rel_sid in same_pool[:extra_same_emotion]:
            qrels.append({"query_id": qid, "story_id": int(rel_sid), "relevance": 1})

    # 2) Emotion-intent queries (Option 1: intent + situation hint)
    for _, row in emo_sample.iterrows():
        sid = int(row["story_id"])
        emo = str(row["emotion_label"]).strip()
        sit = str(row["situation"]).strip()

        qid = f"q{qid_counter}"
        qid_counter += 1

        intent = make_emotion_intent_query(emo)
        hint = make_semantic_hint(sit, max_words=8)

        # Make the query natural and unique per story, without using label tokens directly
        if hint:
            qtext = f"{intent}. Context {hint}"
        else:
            qtext = intent

        queries.append({"query_id": qid, "query_text": qtext, "query_type": "emotion_intent"})

        qrels.append({"query_id": qid, "story_id": sid, "relevance": 2})

        same_pool = [x for x in by_emotion.get(emo, []) if x != sid]
        random.shuffle(same_pool)
        for rel_sid in same_pool[:extra_same_emotion]:
            qrels.append({"query_id": qid, "story_id": int(rel_sid), "relevance": 1})
            # NOTE: query_text field in qrels is optional; remove if your evaluator expects only query_id/story_id/relevance

    # Save
    OUT_Q.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(queries).to_csv(OUT_Q, index=False)

    # Ensure qrels has the expected columns (some evaluators expect exactly these 3 columns)
    qrels_df = pd.DataFrame(qrels)
    if "query_text" in qrels_df.columns:
        qrels_df = qrels_df.drop(columns=["query_text"])
    qrels_df.to_csv(OUT_QRELS, index=False)

    print(f"Saved: {OUT_Q} rows: {len(queries)}")
    print(f"Saved: {OUT_QRELS} rows: {len(qrels_df)}")
    print(f"Semantic queries: {n_semantic} | Emotion-intent queries: {n_emotion_intent}")
    print(f"extra_same_emotion per query: {extra_same_emotion}")

if __name__ == "__main__":
    main(n_semantic=100, n_emotion_intent=100, extra_same_emotion=3)



#16th Jan Change all as duplicate queries generation according to natural language emotion-intent queries
# # src/build_eval_set.py #15th jan Replace the entire file according to natural language emotion-intent queries
# import pandas as pd
# import random
# from pathlib import Path

# RAW_PATH = Path("dataset/raw/empathetic_stories.csv")
# OUT_Q = Path("dataset/eval/queries.csv")
# OUT_QRELS = Path("dataset/eval/qrels.csv")

# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)

# def make_semantic_query(situation: str, max_words: int = 12) -> str:
#     """Short semantic query derived from situation text."""
#     if not isinstance(situation, str):
#         return ""
#     text = situation.replace("\n", " ").strip()
#     words = [w for w in text.split() if w]
#     return " ".join(words[:max_words]).strip()

# # Natural-language emotion-intent prompts (avoid exact label as the only signal)
# # IMPORTANT: These can still contain emotion-related words (lonely, anxious, etc.),
# # because baselines also see the same query text. We avoid the trivial "a <label> experience".
# EMO_INTENT_PROMPTS = {
#     "afraid": "a memory about feeling anxious and unsafe, like something bad might happen",
#     "angry": "a memory about feeling furious after being treated unfairly or disrespected",
#     "annoyed": "a memory about being irritated by something small that kept happening",
#     "anticipating": "a memory about waiting for something important and feeling on edge",
#     "anxious": "a memory about worrying a lot and not being able to relax",
#     "apprehensive": "a memory about being uncertain and nervous about what was coming next",
#     "ashamed": "a memory about feeling embarrassed and regretting something you did",
#     "caring": "a memory about supporting someone and wanting to help them feel better",
#     "confident": "a memory about feeling capable and sure you could handle the situation",
#     "content": "a memory about feeling calm, satisfied, and at peace with how things were",
#     "devastated": "a memory about feeling completely crushed after a big loss or setback",
#     "disappointed": "a memory about feeling let down when something didn’t work out",
#     "disgusted": "a memory about feeling strongly uncomfortable with something unpleasant",
#     "embarrassed": "a memory about feeling awkward after a mistake in front of others",
#     "excited": "a memory about feeling thrilled and full of energy about something new",
#     "faithful": "a memory about staying loyal and committed even when things were difficult",
#     "furious": "a memory about intense anger and wanting to react strongly",
#     "grateful": "a memory about appreciating someone’s help or kindness",
#     "guilty": "a memory about regretting something and wishing you could undo it",
#     "hopeful": "a memory about believing things could improve despite challenges",
#     "impressed": "a memory about admiring someone’s skills or a surprising achievement",
#     "jealous": "a memory about wanting what someone else had and feeling bothered by it",
#     "joyful": "a memory about feeling happy and uplifted by a positive moment",
#     "lonely": "a memory about feeling alone or unsupported, even around other people",
#     "nostalgic": "a memory about looking back warmly on the past and missing it",
#     "prepared": "a memory about feeling ready and organized for something important",
#     "proud": "a memory about feeling accomplished after hard work paid off",
#     "sad": "a memory about feeling down and hurt, like you lost something important",
#     "sentimental": "a memory about deep feelings tied to a meaningful person or moment",
#     "surprised": "a memory about something unexpected that changed how you felt",
#     "terrified": "a memory about intense fear, like you were in real danger",
#     "trusting": "a memory about relying on someone and believing they would support you",
# }

# def make_emotion_intent_query(emotion_label: str) -> str:
#     """Create a natural language emotion-intent query for a given label."""
#     emo = (emotion_label or "").strip().lower()
#     return EMO_INTENT_PROMPTS.get(
#         emo,
#         "a memory about strong emotions and how you handled the situation"
#     )

# def main(
#     n_semantic: int = 100,
#     n_emotion_intent: int = 100,
#     extra_same_emotion: int = 3,
#     max_emotion_pool_scan: int | None = None,
# ):
#     df = pd.read_csv(RAW_PATH)

#     required_cols = {"story_id", "emotion_label", "situation"}
#     missing = required_cols - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing required columns in {RAW_PATH}: {missing}")

#     df["story_id"] = df["story_id"].astype(int)
#     df["emotion_label"] = df["emotion_label"].astype(str).str.strip()
#     df["situation"] = df["situation"].astype(str)

#     # Build emotion -> story_id pool (used ONLY for evaluation/qrels construction)
#     pool_df = df if max_emotion_pool_scan is None else df.head(max_emotion_pool_scan)
#     by_emotion: dict[str, list[int]] = {}
#     for _, r in pool_df.iterrows():
#         emo = str(r["emotion_label"]).strip()
#         sid = int(r["story_id"])
#         by_emotion.setdefault(emo, []).append(sid)

#     # Sample stories for semantic + emotion-intent sets (can overlap, but better if disjoint)
#     n_semantic = min(n_semantic, len(df))
#     n_emotion_intent = min(n_emotion_intent, len(df) - n_semantic)

#     sem_sample = df.sample(n=n_semantic, random_state=RANDOM_SEED).reset_index(drop=True)

#     # remove semantic-picked rows, then sample emotion-intent from remaining for diversity
#     remaining = df[~df["story_id"].isin(sem_sample["story_id"])].copy()
#     emo_sample = remaining.sample(n=n_emotion_intent, random_state=RANDOM_SEED).reset_index(drop=True)

#     queries = []
#     qrels = []
#     qid_counter = 1

#     # -------------------
#     # 1) Semantic queries (100)
#     # -------------------
#     for _, row in sem_sample.iterrows():
#         sid = int(row["story_id"])
#         emo = str(row["emotion_label"]).strip()
#         sit = str(row["situation"]).strip()

#         qid = f"q{qid_counter}"
#         qid_counter += 1

#         qtext = make_semantic_query(sit)
#         if not qtext:
#             qtext = f"a personal situation where someone feels {emo}"

#         queries.append({"query_id": qid, "query_text": qtext, "query_type": "semantic"})

#         # qrels: exact story is relevance 2
#         qrels.append({"query_id": qid, "story_id": sid, "relevance": 2})

#         # add same-emotion docs as relevance 1 (graded relevance)
#         same_pool = [x for x in by_emotion.get(emo, []) if x != sid]
#         random.shuffle(same_pool)
#         for rel_sid in same_pool[:extra_same_emotion]:
#             qrels.append({"query_id": qid, "story_id": int(rel_sid), "relevance": 1})

#     # -------------------
#     # 2) Emotion-intent queries (100) - NATURAL LANGUAGE
#     # -------------------
#     for _, row in emo_sample.iterrows():
#         sid = int(row["story_id"])
#         emo = str(row["emotion_label"]).strip()

#         qid = f"q{qid_counter}"
#         qid_counter += 1

#         qtext = make_emotion_intent_query(emo)
#         queries.append({"query_id": qid, "query_text": qtext, "query_type": "emotion_intent"})

#         qrels.append({"query_id": qid, "story_id": sid, "relevance": 2})

#         same_pool = [x for x in by_emotion.get(emo, []) if x != sid]
#         random.shuffle(same_pool)
#         for rel_sid in same_pool[:extra_same_emotion]:
#             qrels.append({"query_id": qid, "story_id": int(rel_sid), "relevance": 1})

#     OUT_Q.parent.mkdir(parents=True, exist_ok=True)
#     pd.DataFrame(queries).to_csv(OUT_Q, index=False)
#     pd.DataFrame(qrels).to_csv(OUT_QRELS, index=False)

#     print(f"Saved: {OUT_Q} rows: {len(queries)}")
#     print(f"Saved: {OUT_QRELS} rows: {len(qrels)}")
#     print(f"Semantic queries: {n_semantic} | Emotion-intent queries: {n_emotion_intent}")
#     print(f"extra_same_emotion per query: {extra_same_emotion}")

# if __name__ == "__main__":
#     main(n_semantic=100, n_emotion_intent=100, extra_same_emotion=3)
