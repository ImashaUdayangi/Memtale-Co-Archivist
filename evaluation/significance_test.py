# evaluation/significance_test.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["P@5", "nDCG@5", "MRR"]

def paired_wilcoxon(a: np.ndarray, b: np.ndarray):
    diff = a - b
    if np.allclose(diff, 0):
        return None
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(p)

def run_tests(df: pd.DataFrame, ours_name: str, baselines: list[str], group_cols: list[str]):
    rows = []
    for group_key, g in df.groupby(group_cols) if group_cols else [(("ALL",), df)]:
        gkey = group_key if isinstance(group_key, tuple) else (group_key,)
        g_ours = g[g["system"] == ours_name].sort_values("query_id")

        for base in baselines:
            g_base = g[g["system"] == base].sort_values("query_id")

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
                        help="Run folder containing per_query_results.csv (e.g., evaluation/runs/index_sit_aware/alpha_0.9)")
    # 22nd UPDATED 
    parser.add_argument("--ours", type=str, default="Ours_Sem+Emo_sit_aware")

    # 22nd ADDED
    parser.add_argument("--query_type", type=str, default=None,
                        help="Optional filter, e.g., emotion_intent or semantic")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    in_path = run_dir / "per_query_results.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find: {in_path}")

    df = pd.read_csv(in_path)

    # 22nd ADDED NEW filter
    if args.query_type is not None:
        df = df[df["query_type"] == args.query_type].copy()

    systems = sorted(df["system"].unique().tolist())
    baselines = [s for s in systems if s != args.ours]

    out_dir = run_dir / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = run_tests(df, args.ours, baselines, group_cols=[])
    overall.to_csv(out_dir / "wilcoxon_overall.csv", index=False)

    by_type = run_tests(df, args.ours, baselines, group_cols=["query_type"]) if args.query_type is None else pd.DataFrame()
    if not by_type.empty:
        by_type.to_csv(out_dir / "wilcoxon_by_query_type.csv", index=False)

    print(f"Saved:\n- {out_dir / 'wilcoxon_overall.csv'}")
    if args.query_type is None:
        print(f"- {out_dir / 'wilcoxon_by_query_type.csv'}")

    show = overall.sort_values(["metric", "p_value"], na_position="last").head(12)
    print("\nTop comparisons (lowest p-values):")
    print(show.to_string(index=False))

if __name__ == "__main__":
    main()
