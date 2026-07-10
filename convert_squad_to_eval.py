import json
from pathlib import Path


IN_PATH = Path("data/squad_metadata.jsonl")
OUT_PATH = Path("data\eval_set_v0_raw.jsonl")
START_SAMPLE_ID = 43


def load_existing_eval_records(path):
    """
    Load existing questions and gold_chunk_ids from eval_set_v0_raw.jsonl
    so we do not add duplicates.
    """
    existing_questions = set()
    existing_chunk_ids = set()

    if not path.exists():
        return existing_questions, existing_chunk_ids

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                question = record.get("question", "").strip().lower()
                if question:
                    existing_questions.add(question)

                gold_chunk_id = record.get("gold_chunk_id")
                if isinstance(gold_chunk_id, int):
                    existing_chunk_ids.add(gold_chunk_id)

            except json.JSONDecodeError:
                print("Skipping bad JSON line in existing eval file.")

    return existing_questions, existing_chunk_ids


def convert_squad_record(record):
    """
    Convert one SQuAD-style record into eval format.

    Input:
    {
      "sample_id": 99,
      "doc_id": 98,
      "title": "...",
      "question": "...",
      "answers": ["..."],
      "is_impossible": false
    }

    Output:
    {
      "question": "...",
      "answer": "...",
      "gold_chunk_id": 98
    }
    """

    sample_id = record.get("sample_id")

    if not isinstance(sample_id, int):
        return None, "missing_or_invalid_sample_id"

    if sample_id < START_SAMPLE_ID:
        return None, "before_start_sample_id"

    if record.get("is_impossible") is True:
        return None, "is_impossible_true"

    question = record.get("question", "").strip()
    answers = record.get("answers", [])

    if not question:
        return None, "missing_question"

    if not isinstance(answers, list) or len(answers) == 0:
        return None, "missing_answers"

    answer = str(answers[0]).strip()

    if not answer:
        return None, "empty_answer"

    gold_chunk_id = record.get("doc_id")

    if not isinstance(gold_chunk_id, int):
        return None, "missing_or_invalid_doc_id"

    return {
        "question": question,
        "answer": answer,
        "gold_chunk_id": gold_chunk_id
    }, None


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_questions, existing_chunk_ids = load_existing_eval_records(OUT_PATH)

    read_count = 0
    written_count = 0
    skipped_count = 0

    skip_reasons = {}

    with IN_PATH.open("r", encoding="utf-8") as in_file, OUT_PATH.open("a", encoding="utf-8") as out_file:
        for line in in_file:
            if not line.strip():
                continue

            read_count += 1

            try:
                squad_record = json.loads(line)
            except json.JSONDecodeError:
                skipped_count += 1
                skip_reasons["bad_json"] = skip_reasons.get("bad_json", 0) + 1
                continue

            eval_record, reason = convert_squad_record(squad_record)

            if eval_record is None:
                skipped_count += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            normalized_question = eval_record["question"].strip().lower()
            gold_chunk_id = eval_record["gold_chunk_id"]

            if normalized_question in existing_questions:
                skipped_count += 1
                skip_reasons["duplicate_question"] = skip_reasons.get("duplicate_question", 0) + 1
                continue

            if gold_chunk_id in existing_chunk_ids:
                skipped_count += 1
                skip_reasons["duplicate_gold_chunk_id"] = skip_reasons.get("duplicate_gold_chunk_id", 0) + 1
                continue

            out_file.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            out_file.flush()

            existing_questions.add(normalized_question)
            existing_chunk_ids.add(gold_chunk_id)

            written_count += 1

    print("Done.")
    print(f"Input file: {IN_PATH}")
    print(f"Output file: {OUT_PATH}")
    print(f"Records read: {read_count}")
    print(f"Records written: {written_count}")
    print(f"Records skipped: {skipped_count}")

    print("\nSkip reasons:")
    for reason, count in sorted(skip_reasons.items()):
        print(f"- {reason}: {count}")


if __name__ == "__main__":
    main()