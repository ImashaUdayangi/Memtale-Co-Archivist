# Memtale Prototype — Emotion-Aware Summarisation and Retrieval

This repository contains a **fully runnable research prototype** of the *Memtale* system for **emotion-aware summarisation and retrieval of autobiographical-style narratives**.

The prototype demonstrates how **emotional alignment** can be incorporated as an explicit ranking signal on top of standard information retrieval pipelines, using a transparent two-stage architecture.

---

## 1. System Overview

The prototype implements and compares the following retrieval approaches:

- **BM25** — lexical retrieval baseline  
- **SBERT** — dense semantic retrieval baseline  
- **RRF** — hybrid retrieval using Reciprocal Rank Fusion (BM25 + SBERT)  
- **OURS** — proposed two-stage retrieval:

  **BM25 candidate generation → semantic + emotion-aware reranking (SETL-2)**

The system supports **emotion-intent queries**, reflecting how users (e.g., older adults) recall memories through affective cues rather than purely topical descriptions.

All experiments, metrics, and figures can be reproduced using a **single command**.

---

## 2. Project Structure

Memtale-Prototype/
│
├── src/
│ ├── summarisation.py # Naïve and emotion-aware summarisation
│ ├── setl2.py # Interpretable emotion representation (SETL-2)
│ ├── retrieval.py # BM25, SBERT, RRF, and two-stage reranking
│ ├── data_build_empathetic.py # Dataset construction
│ ├── precompute_summaries.py # Cached summaries
│ ├── build_eval_queries_v3.py # Emotion-intent query generation
│ ├── build_judging_pool_v3.py # Judging pool creation (manual step)
│ └── convert_judged_to_qrels_v3.py
│
├── evaluation/
│ ├── run_two_stage_retrieval.py
│ ├── compute_metrics.py
│ ├── compute_emotional_fidelity.py
│ ├── compute_significance_test.py
│ ├── make_main_results_figure.py
│ └── make_qualitative_rerank_example.py
│
├── dataset/
│ ├── raw/empathetic_stories.csv # 5,000 autobiographical-style narratives
│ ├── cache/summaries.csv # Cached summaries
│ └── eval_v3/
│ ├── queries.csv # Emotion-intent queries
│ └── qrels.csv # Manual relevance judgments
│
├── main.py # Single entry point (prototype runner)
├── requirements.txt
└── README.md

---

## 3. Environment Setup

### Requirements
- Python **3.10**
- Tested on **Windows (PowerShell)**

### Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

---

## 4. Install dependencies

pip install -r requirements.txt

Note: The virtual environment (venv/) is not included in the repository and should be created locally.

---

## 5. Running the Prototype

After activating the virtual environment, run:

```powershell
python main.py```

This single command:

Runs all retrieval methods (BM25, SBERT, RRF, OURS)

Computes ranking and emotional alignment metrics

Performs statistical significance testing

Generates plots and qualitative examples

No manual annotation is required.

---

## 6. Outputs

Successful execution produces the following key outputs:

evaluation/results/evaluation_summary.csv
→ nDCG@5 and MRR for all methods

evaluation/results/emotional_fidelity_summary.csv
→ Emotional alignment at early ranks

evaluation/results/significance_summary.csv
→ Paired Wilcoxon signed-rank test results

evaluation/results/main_results_bar_chart.png
→ Combined grouped bar chart (ranking + emotion alignment)

evaluation/results/qualitative_rerank_example.csv
→ Automatically selected reranking illustration

---

## 7. Reproducibility Notes

All datasets, summaries, queries, and relevance judgments are pre-generated and frozen

Manual judging is not required to reproduce results

The prototype is designed for evaluation, demonstration, and inspection

The --rebuild mode (developer-only) regenerates datasets and requires manual relevance annotation

---