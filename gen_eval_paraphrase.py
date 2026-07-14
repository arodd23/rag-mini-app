"""
gen_eval_paraphrase.py

Generates type="paraphrase" eval items: takes existing "answerable" or
"multihop" questions already in the eval file and asks the model to reword
them (different phrasing, same meaning), keeping the same answer and the
same gold_chunk_ids / gold_spans. Does not touch the chunk index or source
docs at all -- it only reads/writes the eval file.

Tracks which source questions have already been paraphrased via a
"source_question" metadata field on each paraphrase record, so re-running
this script tops up rather than re-paraphrasing the same items.

Usage:
    python gen_eval_paraphrase.py
    python gen_eval_paraphrase.py --num-items 40
    python gen_eval_paraphrase.py --source-types answerable
"""

import argparse
import random
import time
from pathlib import Path

from eval_common import get_gemini_client, call_gemini_json, write_record, load_existing_records

DEFAULT_PATH = Path("data/eval/eval_set_v2_raw.jsonl")
DEFAULT_NUM_ITEMS = 40
DEFAULT_SLEEP_SECONDS = 25
DEFAULT_SOURCE_TYPES = "answerable,multihop"

PROMPT_TEMPLATE = """
You are creating a paraphrase for a RAG system evaluation dataset.

Reword the question below so it asks for the exact same information in
different phrasing (different words/sentence structure, same meaning, same
correct answer). Do not change what is being asked. Do not make it more or
less specific.

Original question:
{question}

Rules:
- Return ONLY valid JSON.
- Use exactly this key: "question".

Return JSON:
"""


def load_source_records(all_records, source_types, already_paraphrased_sources):
    pool = []
    for r in all_records:
        if r.get("type") not in source_types:
            continue
        q = r.get("question", "").strip().lower()
        if not q or q in already_paraphrased_sources:
            continue
        pool.append(r)
    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--num-items", type=int, default=DEFAULT_NUM_ITEMS)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument(
        "--source-types",
        default=DEFAULT_SOURCE_TYPES,
        help="comma-separated record types eligible to be paraphrased",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    path = Path(args.path)
    source_types = set(t.strip() for t in args.source_types.split(",") if t.strip())

    existing_questions, _used_chunk_ids, all_records = load_existing_records(path)
    print(f"Total records in {path.name}: {len(all_records)}")

    already_paraphrased_sources = {
        r["source_question"] for r in all_records
        if r.get("type") == "paraphrase" and "source_question" in r
    }
    print(f"Sources already paraphrased: {len(already_paraphrased_sources)}")

    candidates = load_source_records(all_records, source_types, already_paraphrased_sources)
    print(f"Candidate source questions ({', '.join(sorted(source_types))}): {len(candidates)}")

    if not candidates:
        print("No eligible source questions to paraphrase. Nothing to generate.")
        return

    sample_size = min(args.num_items, len(candidates))
    sampled = random.sample(candidates, sample_size)

    client = get_gemini_client()

    written = failed = skipped_duplicates = 0

    with path.open("a", encoding="utf-8") as f:
        for idx, source in enumerate(sampled, start=1):
            source_question = source["question"].strip()
            source_question_norm = source_question.lower()
            print(f"[{idx}/{sample_size}] paraphrasing: {source_question[:80]}...")

            try:
                result = call_gemini_json(
                    client, PROMPT_TEMPLATE.format(question=source_question), args.sleep
                )
                record = {
                    "type": "paraphrase",
                    "question": result.get("question", ""),
                    "answer": source["answer"],
                    "gold_chunk_ids": source["gold_chunk_ids"],
                    "gold_spans": source["gold_spans"],
                    "source_question": source_question_norm,
                }

                outcome = write_record(f, record, existing_questions)
                if outcome == "written":
                    written += 1
                elif outcome == "duplicate":
                    print(f"  Duplicate question skipped: {record['question']}")
                    skipped_duplicates += 1
                else:
                    print("  Invalid record, skipped.")
                    failed += 1

            except Exception as e:
                print(f"  Failed to paraphrase: {e}")
                failed += 1

            time.sleep(args.sleep)

    print()
    print("Done.")
    print(f"Wrote: {written}")
    print(f"Failed: {failed}")
    print(f"Skipped duplicates: {skipped_duplicates}")
    print(f"Output file: {path}")


if __name__ == "__main__":
    main()