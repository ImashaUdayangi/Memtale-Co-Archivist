# evaluation/make_paper_plots.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RUNS_DIR = Path("evaluation/runs")
OUT_DIR = Path("evaluation/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_alpha_runs():
    """
    Supports BOTH folder styles you used:
      A) evaluation/runs/alpha_0.5/summary_by_type.csv
      B) evaluation/runs/index_full/alpha_0.8/summary_by_type.csv  (we will ignore these for the alpha sweep)
    """
    rows = []
    for p in RUNS_DIR.glob("alpha_*/summary_by_type.csv"):
        alpha_str = p.parent.name.replace("alpha_", "")
        try:
            alpha = float(alpha_str)
        except ValueError:
            continue

        df = pd.read_csv(p)
        # keep only Ours + emotion row
        hit = df[(df["system"] == "Ours_Sem+Emo_awareSum") & (df["query_type"] == "emotion")]
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


def plot_alpha_ndcg(alpha_df: pd.DataFrame):
    if alpha_df.empty:
        print("No alpha sweep runs found in evaluation/runs/alpha_*/summary_by_type.csv")
        return

    plt.figure()
    plt.plot(alpha_df["alpha"], alpha_df["nDCG@5"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("nDCG@5 (emotion queries) — Ours")
    plt.title("Sensitivity to alpha (Emotion Retrieval)")
    plt.grid(True)

    out_path = OUT_DIR / "alpha_vs_ndcg_emotion.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def load_index_table(alpha: float = 0.8):
    """
    Reads:
      evaluation/runs/index_full/alpha_0.8/summary_overall.csv
      evaluation/runs/index_aware/alpha_0.8/summary_overall.csv
      evaluation/runs/index_sit_aware/alpha_0.8/summary_overall.csv
    Returns a combined table.
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

    # reorder columns nicely
    table = table[["index_mode", "system", "P@5", "nDCG@5", "MRR"]]
    return table


def save_best_systems_table(table: pd.DataFrame):
    if table.empty:
        print("No index tables found.")
        return

    # Best by nDCG@5 per index
    best = (
        table.sort_values(["index_mode", "nDCG@5"], ascending=[True, False])
             .groupby("index_mode", as_index=False)
             .head(1)
             .reset_index(drop=True)
    )

    out_all = OUT_DIR / "table_alpha0.8_all_systems_by_index.csv"
    out_best = OUT_DIR / "table_alpha0.8_best_system_by_index.csv"
    table.to_csv(out_all, index=False)
    best.to_csv(out_best, index=False)

    print("Saved:", out_all)
    print("Saved:", out_best)

    # Also print best table to terminal
    print("\n=== Best system by index (alpha=0.8) ===")
    print(best.to_string(index=False))


def plot_index_comparison_emotion(alpha: float = 0.8):
    """
    Optional Plot 2:
    Bar chart: emotion nDCG@5 for selected systems across indices.
    """
    indices = ["full", "aware", "sit_aware"]
    systems_to_show = ["B2_Sem_full", "B4_RRF_BM25+SBERT", "Ours_Sem+Emo_awareSum"]

    rows = []
    for idx in indices:
        p = RUNS_DIR / f"index_{idx}" / f"alpha_{alpha}" / "summary_by_type.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["query_type"] == "emotion"]
        df = df[df["system"].isin(systems_to_show)]
        df["index_mode"] = idx
        rows.append(df)

    if not rows:
        print("No summary_by_type.csv found for index comparison plot.")
        return

    data = pd.concat(rows, ignore_index=True)

    # pivot to: index_mode x system values
    pivot = data.pivot(index="index_mode", columns="system", values="nDCG@5").reindex(indices)

    plt.figure()
    pivot.plot(kind="bar")
    plt.xlabel("Indexing strategy")
    plt.ylabel("nDCG@5 (emotion queries)")
    plt.title(f"Indexing strategy comparison (alpha={alpha})")
    plt.grid(True, axis="y")

    out_path = OUT_DIR / f"index_comparison_emotion_ndcg_alpha_{alpha}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    # 4A Plot 1: nDCG@5 vs alpha (ours + emotion queries)
    alpha_df = load_alpha_runs()
    if not alpha_df.empty:
        alpha_df.to_csv(OUT_DIR / "alpha_sweep_ours_emotion.csv", index=False)
        print("Saved:", OUT_DIR / "alpha_sweep_ours_emotion.csv")
    plot_alpha_ndcg(alpha_df)

    # 4B Table: alpha=0.8 for each index (full/aware/sit_aware)
    table = load_index_table(alpha=0.8)
    save_best_systems_table(table)

    # Optional Plot 2: indexing comparison bars
    plot_index_comparison_emotion(alpha=0.8)


if __name__ == "__main__":
    main()