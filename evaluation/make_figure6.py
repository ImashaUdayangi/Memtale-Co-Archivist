# evaluation/make_figure6.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.setl2 import setl2_emotion_vector, EMOTIONS  # your implementation

# -------------------------
# Paths (edit if needed)
# -------------------------
SUMMARIES_PATH = "dataset/cache/summaries.csv"
STORIES_PATH = "dataset/raw/empathetic_stories.csv"
FIDELITY_PATH = "evaluation/figures/emotional_fidelity_per_story.csv"
OUT_FIG = "evaluation/figures/figure6_emotionbars.png"

# -------------------------
# Load data
# -------------------------
sums = pd.read_csv(SUMMARIES_PATH)  # story_id, naive_summary, aware_summary
stories = pd.read_csv(STORIES_PATH) # story_id, emotion_label, situation, Story, source
fid = pd.read_csv(FIDELITY_PATH)

# Merge to one table
df = stories.merge(sums, on="story_id", how="inner").merge(fid, on="story_id", how="inner")

# Build "full text" exactly how you treat it in experiments (situation + story)
df["full_text"] = df["situation"].fillna("").astype(str).str.strip() + " " + df["Story"].fillna("").astype(str).str.strip()
df["len_words"] = df["full_text"].str.split().apply(len)

# -------------------------
# Q1-safe story selection rule:
# top-20% by jsd_gain, then closest to median length
# -------------------------
df_pos = df[df["jsd_gain(naive-aware)"] > 0].copy()
if len(df_pos) == 0:
    raise ValueError("No positive jsd_gain stories found. Check fidelity CSV.")

top_n = max(5, int(0.2 * len(df_pos)))
top = df_pos.nlargest(top_n, "jsd_gain(naive-aware)").copy()

median_len = df["len_words"].median()
top["len_dist"] = (top["len_words"] - median_len).abs()

chosen = top.sort_values(["len_dist", "jsd_gain(naive-aware)"], ascending=[True, False]).iloc[0]
story_id = int(chosen["story_id"])

print("Chosen story_id:", story_id)
print("emotion_label:", chosen["emotion_label"])
print("len_words:", chosen["len_words"])
print("cos_gain:", chosen["cos_gain(aware-naive)"])
print("jsd_gain:", chosen["jsd_gain(naive-aware)"])

# Extract texts
full_text = chosen["full_text"]
naive_sum = chosen["naive_summary"]
aware_sum = chosen["aware_summary"]

# -------------------------
# Compute SETL-2 emotion vectors (dict -> aligned arrays)
# -------------------------
def dict_to_vec(d):
    return np.array([d[e] for e in EMOTIONS], dtype=float)

e_full = dict_to_vec(setl2_emotion_vector(full_text))
e_naive = dict_to_vec(setl2_emotion_vector(naive_sum))
e_aware = dict_to_vec(setl2_emotion_vector(aware_sum))

# -------------------------
# Plot bar chart (Matplotlib, clean)
# -------------------------
x = np.arange(len(EMOTIONS))
width = 0.28

plt.figure(figsize=(10, 4))
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
# Print text blocks for paper (paste-ready)
# -------------------------
def shorten(text, max_words=120):
    words = str(text).split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")

print("\n--- FULL STORY (excerpt) ---\n", shorten(full_text, 120))
print("\n--- NAÏVE SUMMARY ---\n", naive_sum)
print("\n--- EMOTION-AWARE SUMMARY ---\n", aware_sum)
