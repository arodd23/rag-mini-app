import json
from pathlib import Path


DATA_DIR = Path("data")
EXPANSION_DIR = DATA_DIR / "eval_expansion"

BASE_PATH = DATA_DIR / "eval_set_v0.jsonl"
PARAPHRASE_PATH = EXPANSION_DIR / "eval_paraphrase_v0.json"
MULTIHOP_PATH = EXPANSION_DIR / "eval_multihop_chunk_v0.json"
UNANSWERABLE_PATH = EXPANSION_DIR / "eval_unanswerable_v0.jsonl"

OUTPUT_PATH = DATA_DIR / "eval_all_v0.jsonl"

def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():
    all_records = []

    # ========================================================
    # ANSWERABLE
    # ========================================================

    answerable = load_jsonl(BASE_PATH)

    for record in answerable:
        all_records.append({
            "question": record["question"],
            "answer": record["answer"],
            "gold_chunk_id": record["gold_chunk_id"],
            "type": "answerable",
        })

    # ========================================================
    # PARAPHRASE
    # ========================================================

    with PARAPHRASE_PATH.open("r", encoding="utf-8") as f:
        paraphrases = json.load(f)

    for record in paraphrases:
        all_records.append({
            "question": record["paraphrased"],
            "answer": record["answer"],
            "gold_chunk_id": record["gold_chunk_id"],
            "type": "paraphrase",
        })

    # ========================================================
    # MULTI-HOP
    # ========================================================

    with MULTIHOP_PATH.open("r", encoding="utf-8") as f:
        multihop = json.load(f)

    for record in multihop:
        all_records.append({
            "question": record["question"],
            "answer": record["answer"],
            "gold_chunk_ids": record["gold_chunk_ids"],
            "type": "multi_hop",
        })

    # ========================================================
    # UNANSWERABLE
    # ========================================================

    unanswerable = load_jsonl(UNANSWERABLE_PATH)

    for record in unanswerable:
        all_records.append({
            "question": record["question"],
            "answer": None,
            "gold_chunk_id": None,
            "type": "unanswerable",
        })

    # ========================================================
    # WRITE COMBINED FILE
    # ========================================================

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # PRINT COUNTS
    # ========================================================

    print()
    print("Evaluation set created")
    print("======================")
    print(f"Answerable:    {len(answerable)}")
    print(f"Paraphrase:    {len(paraphrases)}")
    print(f"Multi-hop:     {len(multihop)}")
    print(f"Unanswerable:  {len(unanswerable)}")
    print("----------------------")
    print(f"Total:         {len(all_records)}")
    print()
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()