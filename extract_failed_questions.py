"""
extract_failed_questions.py

Creates a clean JSONL file containing only evaluation records that failed
retrieval.

By default, a record is considered a retrieval failure when:
    recall_at_5 == 0

Use evaluation_predictions.jsonl as input, not evaluation_cache.jsonl.
The predictions file contains only the latest completed record for each
question, while the cache contains repeated stage snapshots.

Usage:
    python extract_failed_questions.py
    python extract_failed_questions.py --path results/evaluation_predictions.jsonl
    python extract_failed_questions.py --out results/failed_questions.jsonl
    python extract_failed_questions.py --include-imperfect
"""

import argparse
import json
from pathlib import Path

DEFAULT_PATH = Path("results/evaluation_predictions.jsonl")
DEFAULT_OUT_PATH = Path("results/failed_questions.jsonl")


def load_jsonl(path):
    records = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON on line {line_number} of {path}: {e}"
                ) from e

            records.append(record)

    return records


def is_failed(record, include_imperfect=False):
    recall = record.get("recall_at_5")

    if recall is None:
        return False

    if include_imperfect:
        return float(recall) < 1.0

    return float(recall) == 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument(
        "--include-imperfect",
        action="store_true",
        help=(
            "include records with recall_at_5 below 1.0; "
            "default includes only complete retrieval failures"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.path)
    out_path = Path(args.out)

    if not input_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {input_path}")

    records = load_jsonl(input_path)
    failed_records = [
        record
        for record in records
        if is_failed(record, include_imperfect=args.include_imperfect)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for record in failed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    mode = "imperfect retrieval records" if args.include_imperfect else "complete retrieval failures"

    print(f"Loaded records: {len(records)}")
    print(f"Wrote {len(failed_records)} {mode}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()