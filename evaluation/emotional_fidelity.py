# evaluation/emotional_fidelity.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon

from src.setl2 import setl2_emotion_vector


def dict_to_vec(d: dict, keys: list[str]) -> np.ndarray:
    v = np.array([float(d.get(k, 0.0)) for k in keys], dtype=float)
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def safe_prob(v: np.ndarray) -> np.ndarray:
    v = np.maximum(v, 0.0)
    s = float(v.sum())
    if s < 1e-12:
        # uniform fallback
        return np.ones_like(v) / len(v)
    return v / s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="evaluation/figures",
                        help="Where to save CSV outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stories = pd.read_csv("dataset/raw/empathetic_stories.csv")
    summaries = pd.read_csv("dataset/cache/summaries.csv")

    story_ids = stories["story_id"].astype(int).tolist()

    # align summaries to story order
    summaries = summaries.set_index("story_id").loc[story_ids].reset_index()

    full_texts = (stories["situation"].fillna("") + " " + stories["Story"].fillna("")).tolist()
    naive_summaries = summaries["naive_summary"].fillna("").tolist()
    aware_summaries = summaries["aware_summary"].fillna("").tolist()

    # Determine emotion key space from a few samples (stable)
    # If your setl2 always returns same keys, this is safe.
    sample_vec = setl2_emotion_vector("test")
    keys = sorted(sample_vec.keys())

    rows = []
    for sid, full, naive, aware in zip(story_ids, full_texts, naive_summaries, aware_summaries):
        e_full = dict_to_vec(setl2_emotion_vector(full), keys)
        e_naive = dict_to_vec(setl2_emotion_vector(naive), keys)
        e_aware = dict_to_vec(setl2_emotion_vector(aware), keys)

        # cosine (higher better)
        cos_full_naive = cosine(e_full, e_naive)
        cos_full_aware = cosine(e_full, e_aware)

        # JS divergence (lower better) needs probability distributions
        p_full = safe_prob(e_full)
        p_naive = safe_prob(e_naive)
        p_aware = safe_prob(e_aware)

        js_full_naive = float(jensenshannon(p_full, p_naive, base=2.0) ** 2)  # JSD
        js_full_aware = float(jensenshannon(p_full, p_aware, base=2.0) ** 2)

        rows.append({
            "story_id": sid,
            "cos(full,naive)": cos_full_naive,
            "cos(full,aware)": cos_full_aware,
            "jsd(full,naive)": js_full_naive,
            "jsd(full,aware)": js_full_aware,
            "cos_gain(aware-naive)": cos_full_aware - cos_full_naive,
            "jsd_gain(naive-aware)": js_full_naive - js_full_aware
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "emotional_fidelity_per_story.csv", index=False)

    # aggregate
    summary = {
        "cos_full_naive_mean": float(df["cos(full,naive)"].mean()),
        "cos_full_aware_mean": float(df["cos(full,aware)"].mean()),
        "jsd_full_naive_mean": float(df["jsd(full,naive)"].mean()),
        "jsd_full_aware_mean": float(df["jsd(full,aware)"].mean()),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "emotional_fidelity_summary.csv", index=False)

    # significance tests (paired across stories)
    cos_p = wilcoxon(df["cos(full,aware)"], df["cos(full,naive)"], zero_method="wilcox").pvalue
    jsd_p = wilcoxon(df["jsd(full,aware)"], df["jsd(full,naive)"], zero_method="wilcox").pvalue

    sig = pd.DataFrame([{
        "test": "Wilcoxon paired",
        "metric": "cos(full,aware) > cos(full,naive)",
        "p_value": float(cos_p)
    }, {
        "test": "Wilcoxon paired",
        "metric": "jsd(full,aware) < jsd(full,naive)",
        "p_value": float(jsd_p)
    }])
    sig.to_csv(out_dir / "emotional_fidelity_significance.csv", index=False)

    print(f"Saved:\n- {out_dir / 'emotional_fidelity_per_story.csv'}"
          f"\n- {out_dir / 'emotional_fidelity_summary.csv'}"
          f"\n- {out_dir / 'emotional_fidelity_significance.csv'}")

    print("\n=== Emotional Fidelity Summary ===")
    print(pd.DataFrame([summary]).to_string(index=False))

    print("\n=== Significance ===")
    print(sig.to_string(index=False))


if __name__ == "__main__":
    main()