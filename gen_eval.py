"""
gen_eval.py

Generates type="answerable" eval items: one straightforward, single-chunk,
grounded question per sampled chunk. This is the Phase 2 baseline generator.

Safe to re-run repeatedly -- tops up the eval set, skipping chunks/questions
already used.

Usage:
    python gen_eval.py
    python gen_eval.py --num-chunks 150
    python gen_eval.py --dataset data/gov_report_sample_2k.json --out data/eval_set_v2_raw.jsonl
"""

import argparse
import random
import time
from pathlib import Path

from corpus import load_raw_docs
from build_chunk_index import load_chunk_index, DEFAULT_DATASET_PATH
from eval_common import (
    get_gemini_client,
    call_gemini_json,
    write_record,
    load_existing_records,
    chunk_to_span,
)

DEFAULT_OUT_PATH = Path("data/eval/eval_set_v2_raw.jsonl")
DEFAULT_NUM_CHUNKS = 80
DEFAULT_SLEEP_SECONDS = 25

PROMPT_TEMPLATE = """
You are creating an evaluation dataset for a RAG system.

Given the chunk below, write ONE question and ONE answer.

Rules:
- The question must be answerable using ONLY the chunk.
- The answer must be fully supported by the chunk.
- Do not use outside knowledge.
- Do not ask vague questions.
- Do not ask yes/no questions.
- Keep the answer concise.
- Return ONLY valid JSON.
- Use exactly these keys: "question", "answer".

Chunk:
{chunk_text}

Return JSON:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    print("Loading chunk index...")
    chunk_records = load_chunk_index(dataset_path=dataset_path)
    print(f"Total chunks available: {len(chunk_records)}")

    print("Loading documents (for slicing chunk text)...")
    docs = load_raw_docs(dataset_path)
    docs_by_id = {d["doc_id"]: d["text"] for d in docs}

    existing_questions, used_chunk_ids, _ = load_existing_records(out_path)
    print(f"Existing questions in {out_path.name}: {len(existing_questions)}")
    print(f"Chunks already used as gold anywhere in the file: {len(used_chunk_ids)}")

    unused_chunks = [c for c in chunk_records if c["chunk_id"] not in used_chunk_ids]
    print(f"Unused chunks available: {len(unused_chunks)}")

    sample_size = min(args.num_chunks, len(unused_chunks))
    if sample_size == 0:
        print("No unused chunks available. Nothing to generate.")
        return

    sampled_chunks = random.sample(unused_chunks, sample_size)
    client = get_gemini_client()

    written = failed = skipped_duplicates = 0

    with out_path.open("a", encoding="utf-8") as f:
        for idx, chunk in enumerate(sampled_chunks, start=1):
            chunk_text = docs_by_id[chunk["doc_id"]][chunk["start"]:chunk["end"]]
            print(
                f"[{idx}/{sample_size}] doc={chunk['doc_id']} "
                f"chunk={chunk['chunk_id']} ({len(chunk_text)} chars)..."
            )

            try:
                qa = call_gemini_json(
                    client, PROMPT_TEMPLATE.format(chunk_text=chunk_text), args.sleep
                )
                record = {
                    "type": "answerable",
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "gold_chunk_ids": [chunk["chunk_id"]],
                    "gold_spans": [chunk_to_span(chunk)],
                }

                result = write_record(f, record, existing_questions)
                if result == "written":
                    used_chunk_ids.add(chunk["chunk_id"])
                    written += 1
                elif result == "duplicate":
                    print(f"  Duplicate question skipped: {record['question']}")
                    skipped_duplicates += 1
                else:
                    print(f"  Invalid record for chunk {chunk['chunk_id']}")
                    failed += 1

            except Exception as e:
                print(f"  Failed on chunk {chunk['chunk_id']}: {e}")
                failed += 1

            time.sleep(args.sleep)

    print()
    print("Done.")
    print(f"Wrote: {written}")
    print(f"Failed: {failed}")
    print(f"Skipped duplicates: {skipped_duplicates}")
    print(f"Output file: {out_path}")


if __name__ == "__main__":
    main()