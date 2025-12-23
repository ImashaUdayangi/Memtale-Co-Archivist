# src/data_build_empathetic.py
import pandas as pd
from pathlib import Path

IN_PATH = Path("dataset/raw/emotion-emotion_69k.csv")
OUT_PATH = Path("dataset/raw/empathetic_stories.csv")

def clean_text(x: str) -> str:
    x = "" if pd.isna(x) else str(x)
    x = x.replace("\n", " ").replace("\r", " ").strip()
    # collapse extra spaces
    return " ".join(x.split())

def main():
    df = pd.read_csv(IN_PATH)

    # Required columns from your description
    required = ["id", "Situation", "emotion", "empathetic_dialogues", "labels"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

    # Build final schema (stable for all scripts)
    out = pd.DataFrame({
        "story_id": range(1, len(df) + 1),
        "source_id": df["id"],
        "emotion_label": df["emotion"].apply(clean_text),
        "situation": df["Situation"].apply(clean_text),
        "dialogue": df["empathetic_dialogues"].apply(clean_text),
        "response": df["labels"].apply(clean_text),
        "source": "EmpatheticDialogues_Kaggle",
    })

    # Our model input text (context)
    out["context"] = (out["situation"] + " " + out["dialogue"]).str.strip()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} | rows={len(out)}")
    print("Columns:", out.columns.tolist())

if __name__ == "__main__":
    main()