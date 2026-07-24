"""Merge scorecards produced by parallel repo clones.

Example from the parent folder containing clone-a, clone-b, clone-c:
    python merge_scorecards.py clone-a/results/evaluation_scorecard.csv \
        clone-b/results/evaluation_scorecard.csv clone-c/results/evaluation_scorecard.csv \
        --output merged/evaluation_scorecard.csv

Rows are deduplicated by the experiment identity fields. The newest occurrence
wins, which makes rerunning a clone safe.
"""

import argparse
import csv
from pathlib import Path

KEY_FIELDS = (
    "config_name",
    "num_questions",
    "top_k",
    "eval_file_hash",
    "dataset_hash",
)


def row_key(row):
    return tuple(row.get(field, "") for field in KEY_FIELDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="results/evaluation_scorecard.csv")
    args = parser.parse_args()

    rows = {}
    fieldnames = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.exists():
            print(f"Skipping missing scorecard: {path}")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            for row in reader:
                rows[row_key(row)] = row

    if not rows:
        raise SystemExit("No scorecard rows found.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: row_key(r)))
    print(f"Wrote {len(rows)} unique experiment rows to {output}")


if __name__ == "__main__":
    main()
