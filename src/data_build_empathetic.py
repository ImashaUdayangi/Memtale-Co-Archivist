# src/data_build_empathetic.py
import pandas as pd
from pathlib import Path

IN_PATH = Path("dataset/raw/emotion-emotion_69k.csv")              # Kaggle file
OUT_PATH = Path("dataset/raw/empathetic_stories.csv")  # output clean dataset

TARGET_N = 3000
SOURCE_NAME = "EmpatheticDialogues_Kaggle"


def _clean_text(x) -> str:
    """Safe text cleanup for NaNs and spacing."""
    if pd.isna(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def main():
    # 1) Load
    df = pd.read_csv(IN_PATH)

    # 2) Standardize column names (Kaggle sometimes has casing or extra unnamed columns)
    df.columns = [c.strip() for c in df.columns]

    # Drop extra empty columns created by the trailing ",,"
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # Expected columns: id, Situation, emotion, empathetic_dialogues, labels
    # Some Kaggle exports include unnamed columns at the end; ignore them.
    required = {"Situation", "emotion", "empathetic_dialogues", "labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    # 3) Clean key fields
    df["Situation"] = df["Situation"].apply(_clean_text)
    df["emotion"] = df["emotion"].apply(_clean_text).str.lower()
    df["empathetic_dialogues"] = df["empathetic_dialogues"].apply(_clean_text)
    df["labels"] = df["labels"].apply(_clean_text)

    # Remove blank situations or emotions
    df = df[(df["Situation"] != "") & (df["emotion"] != "")].copy()

    # 4) Build per-row "turn" text = dialogue + agent reply
    # This matches requirement: merge empathetic_dialogues + labels into Story
    def make_turn(row):
        d = row["empathetic_dialogues"]
        a = row["labels"]
        if d and a:
            return f"{d} {a}"
        return d or a

    df["turn_text"] = df.apply(make_turn, axis=1)
    df = df[df["turn_text"] != ""].copy()

    # 5) Group rows with the same Situation (+ emotion) into ONE story row
    # Note: Using (Situation, emotion) avoids mixing different emotions for same situation if any exist.
    grouped = (
        df.groupby(["Situation", "emotion"], as_index=False)["turn_text"]
        .apply(lambda s: " ".join(list(s)))
        .rename(columns={"turn_text": "Story"})
    )

    # Rename to target schema
    grouped = grouped.rename(columns={"Situation": "situation", "emotion": "emotion_label"})
    grouped["source"] = SOURCE_NAME

    # 6) Sample 3000 with balanced emotions
    # If some emotions have few examples after grouping, take as many as possible.
    emotions = grouped["emotion_label"].unique().tolist()
    n_emotions = len(emotions)
    per_emotion = max(1, TARGET_N // n_emotions)

    parts = []
    for emo in emotions:
        chunk = grouped[grouped["emotion_label"] == emo]
        if len(chunk) <= per_emotion:
            parts.append(chunk)
        else:
            parts.append(chunk.sample(n=per_emotion, random_state=42))

    sampled = pd.concat(parts, ignore_index=True)

    # If this still below TARGET_N (because some emotions had too few),
    # top up from the remaining pool.
    if len(sampled) < TARGET_N:
        remaining = grouped.merge(sampled, how="left", indicator=True)
        remaining = remaining[remaining["_merge"] == "left_only"].drop(columns=["_merge"])
        need = TARGET_N - len(sampled)
        if len(remaining) > 0:
            topup = remaining.sample(n=min(need, len(remaining)), random_state=42)
            sampled = pd.concat([sampled, topup], ignore_index=True)

    # If exceeded TARGET_N slightly due to rounding, trim
    sampled = sampled.sample(n=min(TARGET_N, len(sampled)), random_state=42).reset_index(drop=True)

    # 7) Add story_id
    sampled.insert(0, "story_id", range(1, len(sampled) + 1))

    # 8) Final column order 
    sampled = sampled[["story_id", "emotion_label", "situation", "Story", "source"]]

    # 9) Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(OUT_PATH, index=False)

    print("✅ Saved:", OUT_PATH)
    print("Rows:", len(sampled))
    print("Unique emotions:", sampled["emotion_label"].nunique())
    print("Example emotions distribution (top 10):")
    print(sampled["emotion_label"].value_counts().head(10))


if __name__ == "__main__":
    main()