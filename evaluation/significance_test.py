# evaluation/significance_test.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = ["P@5", "nDCG@5", "MRR"]


def paired_wilcoxon(a: np.ndarray, b: np.ndarray):
    """
    Paired Wilcoxon signed-rank test.
    Returns p-value. If all diffs are zero, returns None.
    """
    diff = a - b
    if np.allclose(diff, 0):
        return None
    # zero_method='wilcox' ignores zero diffs safely
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(p)


def run_tests(df: pd.DataFrame, ours_name: str, baselines: list[str], group_cols: list[str]):
    """
    For each group (e.g., overall or by query_type), compare Ours vs each baseline for each metric.
    """
    rows = []
    for group_key, g in df.groupby(group_cols) if group_cols else [(("ALL",), df)]:
        gkey = group_key if isinstance(group_key, tuple) else (group_key,)
        g_ours = g[g["system"] == ours_name].sort_values("query_id")

        for base in baselines:
            g_base = g[g["system"] == base].sort_values("query_id")

            # align by query_id (important!)
            merged = g_ours.merge(g_base, on=["query_id"], suffixes=("_ours", "_base"))
            if merged.empty:
                continue

            for m in METRICS:
                a = merged[f"{m}_ours"].to_numpy(dtype=float)
                b = merged[f"{m}_base"].to_numpy(dtype=float)

                p = paired_wilcoxon(a, b)
                rows.append({
                    "group": "|".join(map(str, gkey)),
                    "baseline": base,
                    "metric": m,
                    "ours_mean": float(np.mean(a)),
                    "base_mean": float(np.mean(b)),
                    "mean_diff(ours-base)": float(np.mean(a - b)),
                    "p_value": p
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run folder containing per_query_results.csv (e.g., evaluation/runs/index_full/alpha_0.8)")
    parser.add_argument("--ours", type=str, default="Ours_Sem+Emo_awareSum")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    in_path = run_dir / "per_query_results.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find: {in_path}")

    df = pd.read_csv(in_path)

    # pick baselines that exist in this run
    systems = sorted(df["system"].unique().tolist())
    baselines = [s for s in systems if s != args.ours]

    out_dir = run_dir / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overall tests
    overall = run_tests(df, args.ours, baselines, group_cols=[])
    overall.to_csv(out_dir / "wilcoxon_overall.csv", index=False)

    # By query_type tests
    by_type = run_tests(df, args.ours, baselines, group_cols=["query_type"])
    by_type.to_csv(out_dir / "wilcoxon_by_query_type.csv", index=False)

    print(f"Saved:\n- {out_dir / 'wilcoxon_overall.csv'}\n- {out_dir / 'wilcoxon_by_query_type.csv'}")

    # quick console summary (show the most important comparisons)
    show = overall.sort_values(["metric", "p_value"], na_position="last").head(12)
    print("\nTop comparisons (lowest p-values):")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()