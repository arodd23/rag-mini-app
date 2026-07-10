import json
import os
import random
from pathlib import Path
import time
from dotenv import load_dotenv
from google import genai

from rag_core import RAGSystem


OUT_PATH = Path("data/eval_set_v0_raw.jsonl")
NUM_CHUNKS_TO_SAMPLE = 80
MODEL_NAME = "gemini-2.5-flash"


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


def build_chunks():
    """
    Uses your existing RAGSystem logic to load docs and chunk them.
    This avoids duplicating chunking code.
    """
    rag = RAGSystem()

    raw_text = rag.load_documents()
    chunks = rag.chunk_text(raw_text)

    chunk_records = []

    for i, chunk_text in enumerate(chunks):
        chunk_records.append({
            "chunk_id": i,
            "text": chunk_text
        })

    return chunk_records


def extract_json(text):
    """
    Gemini might return ```json ... ```.
    This cleans that up before json.loads().
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return json.loads(text)


def load_existing_records(path):
    """
    Loads existing questions and gold chunk IDs so we do not write duplicates.
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

                question = record.get("question", "")
                normalized_question = question.strip().lower()

                if normalized_question:
                    existing_questions.add(normalized_question)

                chunk_id = record.get("gold_chunk_id")

                if isinstance(chunk_id, int):
                    existing_chunk_ids.add(chunk_id)

            except json.JSONDecodeError:
                continue

    return existing_questions, existing_chunk_ids


def generate_qa(client, chunk_text):
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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return extract_json(response.text)


def is_valid_record(record):
    required_keys = {"question", "answer", "gold_chunk_id"}

    if set(record.keys()) != required_keys:
        return False

    if not isinstance(record["question"], str):
        return False

    if not isinstance(record["answer"], str):
        return False

    if not isinstance(record["gold_chunk_id"], int):
        return False

    if not record["question"].strip():
        return False

    if not record["answer"].strip():
        return False

    return True


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading and chunking documents...")
    chunks = build_chunks()

    existing_questions, existing_chunk_ids = load_existing_records(OUT_PATH)

    print(f"Total chunks found: {len(chunks)}")
    print(f"Existing questions found: {len(existing_questions)}")
    print(f"Existing chunk IDs found: {len(existing_chunk_ids)}")

    unused_chunks = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in existing_chunk_ids
    ]

    print(f"Unused chunks available: {len(unused_chunks)}")

    sample_size = min(NUM_CHUNKS_TO_SAMPLE, len(unused_chunks))

    if sample_size == 0:
        print("No unused chunks available. Nothing to generate.")
        return

    sampled_chunks = random.sample(unused_chunks, sample_size)

    client = get_gemini_client()

    written = 0
    failed = 0
    skipped_duplicates = 0

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for idx, chunk in enumerate(sampled_chunks, start=1):
            print(f"Generating item {idx}/{sample_size} from chunk {chunk['chunk_id']}...")

            try:
                qa = generate_qa(client, chunk["text"])

                record = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "gold_chunk_id": chunk["chunk_id"]
                }

                if is_valid_record(record):
                    normalized_question = record["question"].strip().lower()
                    gold_chunk_id = record["gold_chunk_id"]

                    if normalized_question in existing_questions:
                        print(f"Duplicate question skipped: {record['question']}")
                        skipped_duplicates += 1

                    elif gold_chunk_id in existing_chunk_ids:
                        print(f"Duplicate chunk skipped: {gold_chunk_id}")
                        skipped_duplicates += 1

                    else:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()

                        existing_questions.add(normalized_question)
                        existing_chunk_ids.add(gold_chunk_id)

                        written += 1
                else:
                    print(f"Invalid record for chunk {chunk['chunk_id']}")
                    failed += 1

            except Exception as e:
                print(f"Failed on chunk {chunk['chunk_id']}: {e}")
                failed += 1

            time.sleep(25)

    print()
    print("Done.")
    print(f"Wrote: {written}")
    print(f"Failed: {failed}")
    print(f"Skipped duplicates: {skipped_duplicates}")
    print(f"Output file: {OUT_PATH}")


if __name__ == "__main__":
    main()