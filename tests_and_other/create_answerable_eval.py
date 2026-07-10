import json
from pathlib import Path


IN_PATH = Path("data/eval_set_v0.jsonl")
OUT_PATH = Path("data/eval_expansion/eval_answerable_v0.jsonl")


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_seen = 0
    total_written = 0
    total_bad_json = 0
    total_bad_records = 0
    seen_questions = set()

    with IN_PATH.open("r", encoding="utf-8") as infile, OUT_PATH.open(
        "w", encoding="utf-8"
    ) as outfile:

        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            total_seen += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping bad JSON on line {line_number}")
                total_bad_json += 1
                continue

            required_keys = {"question", "answer", "gold_chunk_id"}

            if not required_keys.issubset(record.keys()):
                print(f"Skipping bad record on line {line_number}: missing required keys")
                total_bad_records += 1
                continue

            question = record["question"]

            if question in seen_questions:
                print(f"Skipping duplicate question on line {line_number}")
                continue

            seen_questions.add(question)

            eval_record = {
                "question": record["question"],
                "answer": record["answer"],
                "gold_chunk_id": record["gold_chunk_id"],
                "type": "answerable"
            }

            outfile.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            total_written += 1

    print()
    print("Done creating answerable eval file.")
    print(f"Input file: {IN_PATH}")
    print(f"Output file: {OUT_PATH}")
    print()
    print(f"Total records seen: {total_seen}")
    print(f"Written: {total_written}")
    print(f"Bad JSON skipped: {total_bad_json}")
    print(f"Bad records skipped: {total_bad_records}")


if __name__ == "__main__":
    main()