import json
import random
from pathlib import Path


IN_PATH = Path("data/eval_set_v0.jsonl")
OUT_PATH = Path("data/eval_expansion/eval_paraphrase_v0.json")

NUM_TO_SAMPLE = 55
RANDOM_SEED = 42


def load_eval_records(path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []
    bad_json = 0
    bad_records = 0

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping bad JSON on line {line_number}")
                bad_json += 1
                continue

            required_keys = {"question", "answer", "gold_chunk_id"}

            if not required_keys.issubset(record.keys()):
                print(f"Skipping bad record on line {line_number}: missing required keys")
                bad_records += 1
                continue

            records.append(record)

    return records, bad_json, bad_records


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records, bad_json, bad_records = load_eval_records(IN_PATH)

    if len(records) < NUM_TO_SAMPLE:
        raise ValueError(
            f"Not enough records to sample {NUM_TO_SAMPLE}. "
            f"Only found {len(records)} valid records."
        )

    random.seed(RANDOM_SEED)
    sampled_records = random.sample(records, NUM_TO_SAMPLE)

    paraphrase_records = []

    for record in sampled_records:
        paraphrase_record = {
            "question": record["question"],
            "paraphrased": "",
            "answer": record["answer"],
            "gold_chunk_id": record["gold_chunk_id"],
            "type": "paraphrase"
        }

        paraphrase_records.append(paraphrase_record)

    with OUT_PATH.open("w", encoding="utf-8") as outfile:
        json.dump(
            paraphrase_records,
            outfile,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("Done creating paraphrase eval file.")
    print(f"Input file: {IN_PATH}")
    print(f"Output file: {OUT_PATH}")
    print()
    print(f"Valid records loaded: {len(records)}")
    print(f"Randomly sampled: {NUM_TO_SAMPLE}")
    print(f"Bad JSON skipped: {bad_json}")
    print(f"Bad records skipped: {bad_records}")
    print()
    print("Next step: manually fill in the empty 'paraphrased' field for each record.")


if __name__ == "__main__":
    main()