"""
gen_eval.py

Generates the synthetic eval set (Phase 2/3) from the GovReport corpus.

For each of N sampled, not-yet-used chunks: ask Gemini for one question whose
answer is fully contained in that chunk, validate the shape, dedupe against
what's already in the output file, and append.

Safe to re-run repeatedly -- each run tops up the eval set with more items,
skipping chunks/questions you already have. That's how you grow from ~60
items (Phase 2) to ~200 (Phase 3) without redoing work.

Usage:
    python gen_eval.py                          # generate 80 new items
    python gen_eval.py --num-chunks 150
    python gen_eval.py --dataset data/my_sample.json --out data/eval_set_v1_raw.jsonl
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from corpus import load_raw_docs
from build_chunk_index import load_chunk_index, DEFAULT_DATASET_PATH

MODEL_NAME = "gemini-2.5-flash"
DEFAULT_OUT_PATH = Path("data/eval/eval_set_v1_raw.jsonl")
DEFAULT_NUM_CHUNKS = 80
DEFAULT_SLEEP_SECONDS = 25   # spacing between requests, tune to your API tier
MAX_RETRIES = 5


def get_gemini_client():
    load_dotenv()
    # Force the script to ignore any old GOOGLE_API_KEY from Windows/terminal
    os.environ.pop("GOOGLE_API_KEY", None)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Add it to your .env file.")

    print("Using GEMINI_API_KEY")
    print(f"Key ending: ...{api_key[-6:]}")

    return genai.Client(api_key=api_key)


def extract_json(text):
    """Gemini might return ```json ... ```. Strip that before json.loads()."""
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)


def load_existing_records(path):
    """Load existing questions and gold chunk IDs so we do not write duplicates."""
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
            except json.JSONDecodeError:
                continue

            question = record.get("question", "")
            normalized_question = question.strip().lower()
            if normalized_question:
                existing_questions.add(normalized_question)

            chunk_id = record.get("gold_chunk_id")
            if isinstance(chunk_id, int):
                existing_chunk_ids.add(chunk_id)

    return existing_questions, existing_chunk_ids


def is_valid_record(record):
    required_keys = {"question", "answer", "gold_chunk_id"}

    if set(record.keys()) != required_keys:
        return False
    if not isinstance(record["question"], str) or not record["question"].strip():
        return False
    if not isinstance(record["answer"], str) or not record["answer"].strip():
        return False
    if not isinstance(record["gold_chunk_id"], int):
        return False

    return True


def generate_qa(client, chunk_text, sleep_seconds):
    prompt = f"""
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

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return extract_json(response.text)
        except Exception as e:
            last_err = e
            wait = min(90, sleep_seconds * attempt)
            print(f"  API error (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducible sampling")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    print("Loading chunk index (builds/refreshes the cache if needed)...")
    chunk_records = load_chunk_index(dataset_path=dataset_path)
    print(f"Total chunks available: {len(chunk_records)}")

    print("Loading documents (for slicing chunk text)...")
    docs = load_raw_docs(dataset_path)
    docs_by_id = {d["doc_id"]: d["text"] for d in docs}

    existing_questions, existing_chunk_ids = load_existing_records(out_path)
    print(f"Existing questions in {out_path.name}: {len(existing_questions)}")
    print(f"Existing gold chunk ids: {len(existing_chunk_ids)}")

    unused_chunks = [c for c in chunk_records if c["chunk_id"] not in existing_chunk_ids]
    print(f"Unused chunks available: {len(unused_chunks)}")

    sample_size = min(args.num_chunks, len(unused_chunks))
    if sample_size == 0:
        print("No unused chunks available. Nothing to generate.")
        return

    sampled_chunks = random.sample(unused_chunks, sample_size)
    client = get_gemini_client()

    written = 0
    failed = 0
    skipped_duplicates = 0

    with out_path.open("a", encoding="utf-8") as f:
        for idx, chunk in enumerate(sampled_chunks, start=1):
            chunk_text = docs_by_id[chunk["doc_id"]][chunk["start"]:chunk["end"]]
            print(
                f"[{idx}/{sample_size}] doc={chunk['doc_id']} "
                f"chunk={chunk['chunk_id']} ({len(chunk_text)} chars)..."
            )

            try:
                qa = generate_qa(client, chunk_text, args.sleep)

                record = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "gold_chunk_id": chunk["chunk_id"],
                }

                if is_valid_record(record):
                    normalized_question = record["question"].strip().lower()
                    gold_chunk_id = record["gold_chunk_id"]

                    if normalized_question in existing_questions:
                        print(f"  Duplicate question skipped: {record['question']}")
                        skipped_duplicates += 1
                    elif gold_chunk_id in existing_chunk_ids:
                        print(f"  Duplicate chunk skipped: {gold_chunk_id}")
                        skipped_duplicates += 1
                    else:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        existing_questions.add(normalized_question)
                        existing_chunk_ids.add(gold_chunk_id)
                        written += 1
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