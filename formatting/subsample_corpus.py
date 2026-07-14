"""
subsample_corpus.py

Cuts the corpus down to a smaller, faster-to-iterate-on size. You already
sampled 5,000 docs out of ~19,000 from gov_report; this takes another
(deterministic, seeded) sample of those down to 2,000, which is much closer
to the plan's "500-2,000 docs, fast to iterate on a laptop" guidance.

Usage:
    python subsample_corpus.py --input data/dataset/documents.json \
                                --output data/dataset/gov_report_sample_2k.json \
                                --n 2000
"""

import argparse
import json
import random
from pathlib import Path


def subsample(input_path, output_path, n, seed):
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    before = len(docs)
    docs = [d for d in docs if (d.get("text") or "").strip()]
    dropped_empty = before - len(docs)

    random.seed(seed)
    if len(docs) <= n:
        sampled = docs
        print(f"Corpus already has {len(docs)} usable docs (<= {n}), using all of them.")
    else:
        sampled = random.sample(docs, n)

    sampled.sort(key=lambda d: d["doc_id"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"Loaded {before} docs ({dropped_empty} dropped for empty text).")
    print(f"Sampled {len(sampled)} docs (seed={seed}) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/dataset/gov_report_sample_2k.json")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subsample(args.input, args.output, args.n, args.seed)