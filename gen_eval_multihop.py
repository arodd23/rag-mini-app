"""
gen_eval_multihop.py

Generates type="multihop" eval items: samples a PAIR of chunks from the same
source document and asks the model to write one question that requires
information from both chunks to answer, with gold_chunk_ids = [id_a, id_b]
and gold_spans = [span_a, span_b].

Pairs are drawn from the same doc_id (not across documents) -- in this
corpus (GovReport: independent government reports) two spans of the SAME
report are far more likely to relate to each other than spans from two
unrelated reports, so this gives the model a real chance at writing a
coherent combined question.

Tracks already-used pairs (by exact chunk-id pair) via existing multihop
records, so re-running tops up rather than duplicating exact pairs.

Usage:
    python gen_eval_multihop.py
    python gen_eval_multihop.py --num-pairs 30
"""

import argparse
import random
import time
from collections import defaultdict
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
DEFAULT_NUM_PAIRS = 30
DEFAULT_SLEEP_SECONDS = 25
MIN_CHUNKS_PER_DOC_TO_PAIR = 2

PROMPT_TEMPLATE = """
You are creating an evaluation dataset for a RAG system that tests
multi-hop reasoning: retrieving and combining information from two
different passages of the same document.

You are given two passages, Chunk A and Chunk B, from the same report.

Write ONE question that:
- Can only be fully answered by combining information from BOTH Chunk A and Chunk B.
- Would NOT be fully answerable from either chunk alone.
- Has a concise answer that is fully supported by the two chunks together.
- Does not use outside knowledge.
- Is not a yes/no question.

Chunk A:
{chunk_a}

Chunk B:
{chunk_b}

Rules:
- Return ONLY valid JSON.
- Use exactly these keys: "question", "answer".

Return JSON:
"""


def load_used_pairs(all_records):
    used = set()
    for r in all_records:
        if r.get("type") != "multihop":
            continue
        gci = r.get("gold_chunk_ids", [])
        if isinstance(gci, list) and len(gci) == 2:
            used.add(frozenset(gci))
    return used


def build_candidate_pairs(chunk_records, used_pairs, num_pairs_needed):
    """Group chunks by doc, then sample random within-doc pairs, avoiding repeats."""
    by_doc = defaultdict(list)
    for c in chunk_records:
        by_doc[c["doc_id"]].append(c)

    eligible_docs = [doc_id for doc_id, chunks in by_doc.items() if len(chunks) >= MIN_CHUNKS_PER_DOC_TO_PAIR]
    random.shuffle(eligible_docs)

    pairs = []
    seen_in_this_run = set()

    for doc_id in eligible_docs:
        if len(pairs) >= num_pairs_needed:
            break
        chunks = by_doc[doc_id]
        a, b = random.sample(chunks, 2)
        key = frozenset([a["chunk_id"], b["chunk_id"]])
        if key in used_pairs or key in seen_in_this_run:
            continue
        seen_in_this_run.add(key)
        pairs.append((a, b))

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--num-pairs", type=int, default=DEFAULT_NUM_PAIRS)
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
    used_pairs = load_used_pairs(all_records)
    print(f"Existing multihop pairs already used: {len(used_pairs)}")

    pairs = build_candidate_pairs(chunk_records, used_pairs, args.num_pairs)
    print(f"Sampled {len(pairs)} candidate pairs (requested {args.num_pairs}).")

    if not pairs:
        print("Could not build any candidate pairs. Nothing to generate.")
        return

    client = get_gemini_client()

    written = failed = skipped_duplicates = 0

    with out_path.open("a", encoding="utf-8") as f:
        for idx, (chunk_a, chunk_b) in enumerate(pairs, start=1):
            text_a = docs_by_id[chunk_a["doc_id"]][chunk_a["start"]:chunk_a["end"]]
            text_b = docs_by_id[chunk_b["doc_id"]][chunk_b["start"]:chunk_b["end"]]
            print(
                f"[{idx}/{len(pairs)}] doc={chunk_a['doc_id']} "
                f"chunks=({chunk_a['chunk_id']}, {chunk_b['chunk_id']})..."
            )

            try:
                qa = call_gemini_json(
                    client,
                    PROMPT_TEMPLATE.format(chunk_a=text_a, chunk_b=text_b),
                    args.sleep,
                )
                record = {
                    "type": "multihop",
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "gold_chunk_ids": [chunk_a["chunk_id"], chunk_b["chunk_id"]],
                    "gold_spans": [chunk_to_span(chunk_a), chunk_to_span(chunk_b)],
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
                print(f"  Failed on pair ({chunk_a['chunk_id']}, {chunk_b['chunk_id']}): {e}")
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