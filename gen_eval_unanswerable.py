"""
gen_eval_unanswerable.py

Generates type="unanswerable" eval items: seeds from a real chunk (so the
question is topically plausible for this corpus) but explicitly asks the
model for a question that chunk does NOT answer -- e.g. it asks for a
specific figure, date, name, or detail the chunk doesn't state.

gold_chunk_ids and gold_spans are always [] for this type, since correct
behavior is to retrieve nothing / abstain. The "answer" field is a fixed
abstention string rather than model output, since there is nothing in the
corpus to cite.

The seed chunk is stored as "seed_span" (doc_id/start/end -- NOT a chunk id,
since chunk ids go stale across rechunking the same way gold_chunk_ids do).
It has no retrieval meaning; it's only there so re-running this script can
avoid reusing the same seed passage.

Usage:
    python gen_eval_unanswerable.py
    python gen_eval_unanswerable.py --num-items 20
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
    load_used_seed_spans,
    chunk_to_span,
    ABSTENTION_ANSWER,
)

DEFAULT_OUT_PATH = Path("data/eval/eval_set_v2_raw.jsonl")
DEFAULT_NUM_ITEMS = 20
DEFAULT_SLEEP_SECONDS = 25

PROMPT_TEMPLATE = """
You are creating an evaluation dataset for a RAG system's ability to
recognize when it should say "I don't know."

You are given one passage from a government report below.

Write ONE question that:
- Sounds like a natural, specific question someone might ask about this
  general topic or report.
- Is CLOSELY related to the subject matter of the passage (same topic,
  same document, same general area).
- Is NOT answerable from the passage -- it should ask for a specific fact,
  figure, date, name, or detail that this passage does not state.
- Is not obviously a trick question -- it should look like a legitimate
  query someone unfamiliar with exactly what this passage covers might ask.

Passage:
{chunk_text}

Rules:
- Return ONLY valid JSON.
- Use exactly this key: "question".

Return JSON:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--num-items", type=int, default=DEFAULT_NUM_ITEMS)
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

    existing_questions, _used_chunk_ids, all_records = load_existing_records(out_path)
    used_seed_spans = load_used_seed_spans(all_records)
    print(f"Seed spans already used for unanswerable items: {len(used_seed_spans)}")

    unused_chunks = [
        c for c in chunk_records
        if (c["doc_id"], c["start"], c["end"]) not in used_seed_spans
    ]
    print(f"Unused seed chunks available: {len(unused_chunks)}")

    sample_size = min(args.num_items, len(unused_chunks))
    if sample_size == 0:
        print("No unused seed chunks available. Nothing to generate.")
        return

    sampled_chunks = random.sample(unused_chunks, sample_size)
    client = get_gemini_client()

    written = failed = skipped_duplicates = 0

    with out_path.open("a", encoding="utf-8") as f:
        for idx, chunk in enumerate(sampled_chunks, start=1):
            chunk_text = docs_by_id[chunk["doc_id"]][chunk["start"]:chunk["end"]]
            print(
                f"[{idx}/{sample_size}] doc={chunk['doc_id']} "
                f"seed_chunk={chunk['chunk_id']} ({len(chunk_text)} chars)..."
            )

            try:
                result = call_gemini_json(
                    client, PROMPT_TEMPLATE.format(chunk_text=chunk_text), args.sleep
                )
                record = {
                    "type": "unanswerable",
                    "question": result.get("question", ""),
                    "answer": ABSTENTION_ANSWER,
                    "gold_chunk_ids": [],
                    "gold_spans": [],
                    "seed_span": chunk_to_span(chunk),
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
                print(f"  Failed on seed chunk {chunk['chunk_id']}: {e}")
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