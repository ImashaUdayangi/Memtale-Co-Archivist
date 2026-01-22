# evaluation/make_paper_plots.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = Path("evaluation/runs")
OUT_DIR = Path("evaluation/figures_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_ALPHA = 0.9
ALPHA_SWEEP_INDEX = "sit_aware"   # where your alpha sweep was run
TARGET_QUERY_TYPE = "emotion_intent"


def load_alpha_runs(index_mode: str = ALPHA_SWEEP_INDEX):
    """
    Loads alpha sweep from:
      evaluation/runs/index_{index_mode}/alpha_0.5/summary_by_type.csv
    Extracts Ours row for query_type = emotion_intent
    """
    rows = []
    base = RUNS_DIR / f"index_{index_mode}"

    for p in base.glob("alpha_*/summary_by_type.csv"):
        alpha_str = p.parent.name.replace("alpha_", "")
        try:
            alpha = float(alpha_str)
        except ValueError:
            continue

        df = pd.read_csv(p)

        # Ours system name depends on index: Ours_Sem+Emo_sit_aware etc.
        ours_name = f"Ours_Sem+Emo_{index_mode}"

        hit = df[(df["system"] == ours_name) & (df["query_type"] == TARGET_QUERY_TYPE)]
        if len(hit) == 0:
            continue

        rows.append({
            "alpha": alpha,
            "P@5": float(hit["P@5"].iloc[0]),
            "nDCG@5": float(hit["nDCG@5"].iloc[0]),
            "MRR": float(hit["MRR"].iloc[0]),
        })

    out = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    return out


def plot_alpha_ndcg(alpha_df: pd.DataFrame, index_mode: str = ALPHA_SWEEP_INDEX):
    if alpha_df.empty:
        print("No alpha sweep runs found.")
        return

    plt.figure()
    plt.plot(alpha_df["alpha"], alpha_df["nDCG@5"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel(f"nDCG@5 ({TARGET_QUERY_TYPE}) — Ours")
    plt.title(f"Alpha sensitivity (index={index_mode})")
    plt.grid(True)

    out_path = OUT_DIR / f"alpha_vs_ndcg_{TARGET_QUERY_TYPE}_index_{index_mode}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def load_index_table(alpha: float = BEST_ALPHA):
    """
    Reads:
      evaluation/runs/index_full/alpha_0.9/summary_overall.csv
      evaluation/runs/index_aware/alpha_0.9/summary_overall.csv
      evaluation/runs/index_sit_aware/alpha_0.9/summary_overall.csv
    """
    indices = ["full", "aware", "sit_aware"]
    all_rows = []

    for idx in indices:
        p = RUNS_DIR / f"index_{idx}" / f"alpha_{alpha}" / "summary_overall.csv"
        if not p.exists():
            print("Missing:", p)
            continue

        df = pd.read_csv(p)
        df["index_mode"] = idx
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    table = pd.concat(all_rows, ignore_index=True)
    table = table[["index_mode", "system", "P@5", "nDCG@5", "MRR"]]
    return table


def save_best_systems_table(table: pd.DataFrame, alpha: float = BEST_ALPHA):
    if table.empty:
        print("No index tables found.")
        return

    best = (
        table.sort_values(["index_mode", "nDCG@5"], ascending=[True, False])
             .groupby("index_mode", as_index=False)
             .head(1)
             .reset_index(drop=True)
    )

    out_all = OUT_DIR / f"table_alpha{alpha}_all_systems_by_index.csv"
    out_best = OUT_DIR / f"table_alpha{alpha}_best_system_by_index.csv"
    table.to_csv(out_all, index=False)
    best.to_csv(out_best, index=False)

    print("Saved:", out_all)
    print("Saved:", out_best)

    print(f"\n=== Best system by index (alpha={alpha}) ===")
    print(best.to_string(index=False))


def plot_index_comparison_emotion(alpha: float = BEST_ALPHA):
    """
    Bar chart: emotion_intent nDCG@5 for key systems across indices.
    """
    indices = ["full", "aware", "sit_aware"]

    rows = []
    for idx in indices:
        p = RUNS_DIR / f"index_{idx}" / f"alpha_{alpha}" / "summary_by_type.csv"
        if not p.exists():
            continue

        df = pd.read_csv(p)
        df = df[df["query_type"] == TARGET_QUERY_TYPE].copy()

        # Keep only these three systems for each index
        keep = {
            f"B2_Sem_{idx}",
            f"B4_RRF_{idx}",
            f"Ours_Sem+Emo_{idx}",
        }
        df = df[df["system"].isin(keep)]
        df["index_mode"] = idx
        rows.append(df)

    if not rows:
        print("No summary_by_type.csv found for index comparison plot.")
        return

    data = pd.concat(rows, ignore_index=True)

    pivot = data.pivot(index="index_mode", columns="system", values="nDCG@5").reindex(indices)

    plt.figure()
    pivot.plot(kind="bar")
    plt.xlabel("Indexing strategy")
    plt.ylabel(f"nDCG@5 ({TARGET_QUERY_TYPE})")
    plt.title(f"Index strategy comparison (alpha={alpha})")
    plt.grid(True, axis="y")

    out_path = OUT_DIR / f"index_comparison_{TARGET_QUERY_TYPE}_ndcg_alpha_{alpha}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    # Plot 1: alpha sweep (ours only) for chosen index
    alpha_df = load_alpha_runs(ALPHA_SWEEP_INDEX)
    if not alpha_df.empty:
        out_csv = OUT_DIR / f"alpha_sweep_ours_{TARGET_QUERY_TYPE}_index_{ALPHA_SWEEP_INDEX}.csv"
        alpha_df.to_csv(out_csv, index=False)
        print("Saved:", out_csv)
    plot_alpha_ndcg(alpha_df, ALPHA_SWEEP_INDEX)

    # Table: overall metrics by index for best alpha
    table = load_index_table(alpha=BEST_ALPHA)
    save_best_systems_table(table, alpha=BEST_ALPHA)

    # Plot 2: index comparison on emotion_intent
    plot_index_comparison_emotion(alpha=BEST_ALPHA)


if __name__ == "__main__":
    main()
