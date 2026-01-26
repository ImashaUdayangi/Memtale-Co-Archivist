# evaluation/make_figure6.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.setl2 import setl2_emotion_vector, EMOTIONS

# -------------------------
# Paths (UPDATED for figures_2)
# -------------------------
SUMMARIES_PATH = "dataset/cache/summaries.csv"
STORIES_PATH   = "dataset/raw/empathetic_stories.csv"

# use the NEW fidelity output (figures_2)
FIDELITY_PATH  = "evaluation/figures_2/emotional_fidelity_per_story.csv"

# save into figures_2 as well
OUT_FIG        = "evaluation/figures_2/figure6_emotionbars.png"


def dict_to_vec(d: dict) -> np.ndarray:
    """Align dict -> vector using fixed EMOTIONS order."""
    return np.array([float(d.get(e, 0.0)) for e in EMOTIONS], dtype=float)


def shorten(text: str, max_words: int = 120) -> str:
    words = str(text).split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def main():
    # -------------------------
    # Load data
    # -------------------------
    sums = pd.read_csv(SUMMARIES_PATH)     # story_id, naive_summary, aware_summary
    stories = pd.read_csv(STORIES_PATH)   # story_id, emotion_label, situation, Story, ...
    fid = pd.read_csv(FIDELITY_PATH)      # story_id, cos/jsd columns, gains

    # Merge to one table
    df = stories.merge(sums, on="story_id", how="inner").merge(fid, on="story_id", how="inner")

    # Build "full text" exactly as used elsewhere: situation + Story
    df["full_text"] = (
        df["situation"].fillna("").astype(str).str.strip()
        + " "
        + df["Story"].fillna("").astype(str).str.strip()
    )
    df["len_words"] = df["full_text"].str.split().apply(len)

    # -------------------------
    # Reproducible selection rule (non-cherry-picked):
    # top-20% by jsd_gain, then closest to median length
    # -------------------------
    gain_col = "jsd_gain(naive-aware)"
    if gain_col not in df.columns:
        raise ValueError(f"Missing column '{gain_col}' in {FIDELITY_PATH}")

    df_pos = df[df[gain_col] > 0].copy()
    if len(df_pos) == 0:
        raise ValueError("No positive jsd_gain stories found. Check emotional_fidelity_per_story.csv")

    # remove potentially sensitive stories (simple lexical filter for paper figure)
    SENSITIVE_TERMS = [
        "sex", "sleeping with", "affair", "cheated", "girlfriend", "boyfriend",
        "wife", "husband", "mom", "mother", "father"
    ]

    mask = ~df_pos["full_text"].str.lower().str.contains(
        "|".join(SENSITIVE_TERMS), regex=True
    )
    df_pos = df_pos[mask]

    top_n = max(5, int(0.2 * len(df_pos)))
    top = df_pos.nlargest(top_n, gain_col).copy()

    median_len = df["len_words"].median()
    top["len_dist"] = (top["len_words"] - median_len).abs()

    chosen = top.sort_values(["len_dist", gain_col], ascending=[True, False]).iloc[0]
    story_id = int(chosen["story_id"])

    print("Chosen story_id:", story_id)
    if "emotion_label" in chosen:
        print("emotion_label:", chosen["emotion_label"])
    print("len_words:", int(chosen["len_words"]))
    print("cos_gain:", float(chosen["cos_gain(aware-naive)"]))
    print("jsd_gain:", float(chosen[gain_col]))

    # Extract texts
    full_text = chosen["full_text"]
    naive_sum = chosen["naive_summary"]
    aware_sum = chosen["aware_summary"]

    # -------------------------
    # Compute SETL-2 emotion vectors (dict -> aligned arrays)
    # -------------------------
    e_full  = dict_to_vec(setl2_emotion_vector(full_text))
    e_naive = dict_to_vec(setl2_emotion_vector(naive_sum))
    e_aware = dict_to_vec(setl2_emotion_vector(aware_sum))

    # Optional: normalize for nicer comparison
    # If setl2 outputs already normalized scores, you can remove this block.
    def safe_norm(v):
        s = float(np.sum(v))
        return v / s if s > 1e-12 else np.ones_like(v) / len(v)

    e_full  = safe_norm(e_full)
    e_naive = safe_norm(e_naive)
    e_aware = safe_norm(e_aware)

    # -------------------------
    # Plot bar chart (Matplotlib)
    # -------------------------
    x = np.arange(len(EMOTIONS))
    width = 0.28

    plt.figure(figsize=(12, 4.5))
    plt.bar(x - width, e_full,  width=width, label="Full story")
    plt.bar(x,         e_naive, width=width, label="Naïve summary")
    plt.bar(x + width, e_aware, width=width, label="Emotion-aware summary")

    plt.xticks(x, EMOTIONS, rotation=30, ha="right")
    plt.ylabel("Normalized emotion score")
    plt.title("Emotion distribution comparison (SETL-2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    print("Saved figure:", OUT_FIG)

    # -------------------------
    # Print paste-ready text blocks for paper appendix / qualitative section
    # -------------------------
    print("\n--- FULL STORY (excerpt) ---\n", shorten(full_text, 120))
    print("\n--- NAÏVE SUMMARY ---\n", naive_sum)
    print("\n--- EMOTION-AWARE SUMMARY ---\n", aware_sum)


if __name__ == "__main__":
    main()
