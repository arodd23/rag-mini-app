import json
from pathlib import Path


IN_PATH = Path("data/squad_metadata.jsonl")
OUT_PATH = Path("data/eval_expansion/eval_unanswerable_v0.jsonl")


MAX_UNANSWERABLE = None


def load_existing_questions(path):
    """
    Prevents duplicates if you run the script more than once.
    """
    existing_questions = set()

    if not path.exists():
        return existing_questions

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = record.get("question")
            if question:
                existing_questions.add(question.strip())

    return existing_questions


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_questions = load_existing_questions(OUT_PATH)

    total_seen = 0
    total_impossible = 0
    total_written = 0
    total_duplicates = 0
    total_bad_json = 0
    total_missing_question = 0

    with IN_PATH.open("r", encoding="utf-8") as infile, OUT_PATH.open(
        "a", encoding="utf-8"
    ) as outfile:

        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            total_seen += 1

            try:
                squad_record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping bad JSON on line {line_number}")
                total_bad_json += 1
                continue

            if squad_record.get("is_impossible") is not True:
                continue

            total_impossible += 1

            question = squad_record.get("question")

            if not question or not question.strip():
                total_missing_question += 1
                continue

            question = question.strip()

            if question in existing_questions:
                total_duplicates += 1
                continue

            eval_record = {
                "question": question,
                "answer": None,
                "gold_chunk_id": None,
                "type": "unanswerable"
            }

            outfile.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            outfile.flush()

            existing_questions.add(question)
            total_written += 1

            if MAX_UNANSWERABLE is not None and total_written >= MAX_UNANSWERABLE:
                break

    print()
    print("Done creating unanswerable eval file.")
    print(f"Input file: {IN_PATH}")
    print(f"Output file: {OUT_PATH}")
    print()
    print(f"Total records seen: {total_seen}")
    print(f"Impossible records found: {total_impossible}")
    print(f"Written: {total_written}")
    print(f"Duplicates skipped: {total_duplicates}")
    print(f"Bad JSON skipped: {total_bad_json}")
    print(f"Missing question skipped: {total_missing_question}")


if __name__ == "__main__":
    main()