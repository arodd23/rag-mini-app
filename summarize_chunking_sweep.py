"""
summarize_chunking_sweep.py

Reads the accumulated results/evaluation_scorecard.csv (one row per
evaluate.py run, appended by config) and produces the Phase 4 deliverable:
a markdown comparison table and a bar chart of chunking strategy/size/
overlap vs recall@5 and faithfulness.

Usage:
    python summarize_chunking_sweep.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

SCORECARD_PATH = Path("results/evaluation_scorecard.csv")
OUT_DIR = Path("reports")


def load_rows():
    with SCORECARD_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORECARD_PATH.exists():
        print(f"No scorecard at {SCORECARD_PATH} yet -- run evaluate.py or run_chunking_sweep.py first.")
        return

    rows = load_rows()
    if not rows:
        print(f"{SCORECARD_PATH} is empty -- nothing to summarize.")
        return

    rows.sort(key=lambda r: (r["strategy"], int(r["chunk_size"]), int(r["overlap"] or 0)))

    labels = [f"{r['strategy']}\n{r['chunk_size']}/{r['overlap']}" for r in rows]
    recall = [to_float(r["recall_at_5"]) or 0 for r in rows]
    faithfulness = [to_float(r["faithfulness"]) or 0 for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 5))
    x = range(len(rows))
    width = 0.35
    ax.bar([i - width / 2 for i in x], recall, width, label="Recall@5")
    ax.bar([i + width / 2 for i in x], faithfulness, width, label="Faithfulness")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Chunking sweep: Recall@5 vs Faithfulness by config")
    fig.tight_layout()

    chart_path = OUT_DIR / "chunking_sweep.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Wrote chart to {chart_path}")

    md_path = OUT_DIR / "chunking.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 4: Chunking sweep results\n\n")
        f.write("| Strategy | Chunk size | Overlap | Recall@5 | MRR | nDCG@5 | Faithfulness | Answer Relevancy | N |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['strategy']} | {r['chunk_size']} | {r['overlap']} "
                f"| {r['recall_at_5']} | {r['mrr']} | {r['ndcg_at_5']} "
                f"| {r['faithfulness']} | {r['answer_relevancy']} | {r['num_questions']} |\n"
            )
    print(f"Wrote table to {md_path}")

    best = max((r for r in rows if to_float(r["recall_at_5"]) is not None), key=lambda r: float(r["recall_at_5"]), default=None)
    if best:
        print(f"\nBest recall@5: {best['config_name']} ({best['recall_at_5']})")


if __name__ == "__main__":
    main()
