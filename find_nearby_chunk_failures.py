"""
find_nearby_chunk_failures.py

Finds failed retrieval questions where at least one retrieved chunk ID is
within a configurable numeric distance of a gold chunk ID.

Default distance:
    +/- 3 chunk IDs

This is a fast diagnostic only. Numeric proximity does NOT prove that the
retrieved chunk contains the answer. Overlapping or adjacent chunks must
still be inspected using their actual text.

Input should be results/failed_questions.jsonl, created from the completed
evaluation_predictions.jsonl file.

Usage:
    python find_nearby_chunk_failures.py
    python find_nearby_chunk_failures.py --distance 3
    python find_nearby_chunk_failures.py --path results/failed_questions.jsonl
    python find_nearby_chunk_failures.py --out results/nearby_chunk_failures.jsonl
"""

import argparse
import json
from pathlib import Path

DEFAULT_PATH = Path("results/failed_questions.jsonl")
DEFAULT_OUT_PATH = Path("results/nearby_chunk_failures.jsonl")
DEFAULT_DISTANCE = 3


def load_jsonl(path):
    records = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON on line {line_number} of {path}: {e}"
                ) from e

    return records


def get_nearby_matches(record, distance):
    gold_ids = [
        int(chunk_id)
        for chunk_id in (record.get("gold_chunk_ids") or [])
    ]

    retrieved_ids = [
        int(chunk_id)
        for chunk_id in (record.get("retrieved_chunk_ids") or [])
    ]

    matches = []

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        for gold_id in gold_ids:
            difference = retrieved_id - gold_id

            if 0 < abs(difference) <= distance:
                matches.append({
                    "gold_chunk_id": gold_id,
                    "retrieved_chunk_id": retrieved_id,
                    "difference": difference,
                    "retrieved_rank": rank,
                })

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument(
        "--distance",
        type=int,
        default=DEFAULT_DISTANCE,
    )
    args = parser.parse_args()

    input_path = Path(args.path)
    out_path = Path(args.out)

    if args.distance < 1:
        raise ValueError("--distance must be at least 1")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Failed-question file not found: {input_path}\n"
            "Run extract_failed_questions.py first."
        )

    records = load_jsonl(input_path)
    nearby_records = []

    for record in records:
        matches = get_nearby_matches(
            record,
            distance=args.distance,
        )

        if not matches:
            continue

        entry = dict(record)
        entry["nearby_chunk_matches"] = matches
        nearby_records.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for record in nearby_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Failed records checked: {len(records)}"
    )
    print(
        f"Questions with a retrieved chunk within "
        f"+/-{args.distance}: {len(nearby_records)}"
    )
    print(f"Output: {out_path}")

    if nearby_records:
        print()
        print("Nearby failures:")

        for record in nearby_records:
            question = record.get("question", "")
            print(f"- {question}")

            for match in record["nearby_chunk_matches"]:
                sign = "+" if match["difference"] > 0 else ""
                print(
                    f"    gold={match['gold_chunk_id']} "
                    f"retrieved={match['retrieved_chunk_id']} "
                    f"diff={sign}{match['difference']} "
                    f"rank={match['retrieved_rank']}"
                )


if __name__ == "__main__":
    main()