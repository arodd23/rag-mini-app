"""
eval_common.py

Shared helpers used by all eval-set generation/review scripts:
    gen_eval.py                (type="answerable")
    gen_eval_paraphrase.py     (type="paraphrase")
    gen_eval_multihop.py       (type="multihop")
    gen_eval_unanswerable.py   (type="unanswerable")
    review_eval_set.py         (automated first-pass QA)

All records share this shape:

    {
        "type": "answerable" | "paraphrase" | "multihop" | "unanswerable",
        "question": str,
        "answer": str,
        "gold_chunk_ids": [int, ...],          # always a list; [] for unanswerable
        "gold_spans": [{"doc_id": int, "start": int, "end": int}, ...]
    }

Why both gold_chunk_ids AND gold_spans:
    gold_chunk_ids refers to chunk ids under the CURRENT chunk_index.jsonl
    (built by build_chunk_index.py with a specific chunk_size/overlap/
    strategy). The moment you rechunk with different settings (Phase 4/5),
    those ids point at different text -- they go stale.

    gold_spans are raw (doc_id, start, end) character offsets into the
    source document. Those never go stale, regardless of how you rechunk.
    THIS is the field evaluate.py should use for recall/MRR/nDCG once you
    start sweeping chunking strategies: a retrieved chunk counts as a hit
    if its character range overlaps a gold span, not if its chunk_id
    matches.

    gold_chunk_ids is kept alongside purely as a convenience for quick
    lookups against the current index -- treat it as disposable.

Question-text dedup is global across the whole file, regardless of type.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 5
VALID_TYPES = {"answerable", "paraphrase", "multihop", "unanswerable"}

# Canned answer stored on unanswerable records -- there IS no answer in the
# corpus, so we don't ask the model to invent one. This mirrors the Phase 8
# abstention behavior the app is eventually supposed to produce.
ABSTENTION_ANSWER = "I do not have enough information to answer that."


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


def call_gemini_json(client, prompt, sleep_seconds, max_retries=MAX_RETRIES):
    """Call Gemini, parse JSON out of the response, retrying with backoff on error."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return extract_json(response.text)
        except Exception as e:
            last_err = e
            wait = min(90, sleep_seconds * attempt)
            print(f"  API error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise last_err


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------

def chunk_to_span(chunk):
    """Convert a chunk_index.jsonl record into a durable (doc_id, start, end) span."""
    return {"doc_id": chunk["doc_id"], "start": chunk["start"], "end": chunk["end"]}


def span_text(docs_by_id, span):
    """Slice the actual text out of a doc for a given span."""
    return docs_by_id[span["doc_id"]][span["start"]:span["end"]]


def _is_valid_span(span):
    if not isinstance(span, dict):
        return False
    if not {"doc_id", "start", "end"} <= span.keys():
        return False
    if not isinstance(span["doc_id"], int):
        return False
    if not isinstance(span["start"], int) or not isinstance(span["end"], int):
        return False
    if span["start"] >= span["end"] or span["start"] < 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Record validation / dedup
# ---------------------------------------------------------------------------

def is_valid_record(record):
    """
    Structural validation shared by all record types.
    Uses issubset (not exact key match) so type-specific scripts can attach
    extra metadata fields (e.g. "seed_span", "source_question") without
    failing validation.
    """
    required = {"type", "question", "answer", "gold_chunk_ids", "gold_spans"}
    if not required.issubset(record.keys()):
        return False

    if record["type"] not in VALID_TYPES:
        return False

    if not isinstance(record["question"], str) or not record["question"].strip():
        return False

    if not isinstance(record["answer"], str) or not record["answer"].strip():
        return False

    gci = record["gold_chunk_ids"]
    if not isinstance(gci, list) or not all(isinstance(x, int) for x in gci):
        return False

    spans = record["gold_spans"]
    if not isinstance(spans, list) or not all(_is_valid_span(s) for s in spans):
        return False

    # unanswerable records must carry no gold evidence; all other types must.
    if record["type"] == "unanswerable":
        if gci or spans:
            return False
    else:
        if not gci or not spans:
            return False

    return True


def load_existing_records(path):
    """
    Scan the (possibly multi-type) eval file.

    Returns:
        existing_questions: set of normalized question strings, across ALL types
        used_chunk_ids: set of every chunk id in any record's gold_chunk_ids
        raw_records: list of the parsed record dicts, in file order
    """
    existing_questions = set()
    used_chunk_ids = set()
    raw_records = []

    path = Path(path)
    if not path.exists():
        return existing_questions, used_chunk_ids, raw_records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_records.append(record)

            q = record.get("question", "").strip().lower()
            if q:
                existing_questions.add(q)

            gci = record.get("gold_chunk_ids", [])
            if isinstance(gci, list):
                for cid in gci:
                    if isinstance(cid, int):
                        used_chunk_ids.add(cid)

    return existing_questions, used_chunk_ids, raw_records


def load_used_seed_spans(all_records):
    """Spans already used to seed an unanswerable question, so we don't reuse them."""
    used = set()
    for r in all_records:
        if r.get("type") != "unanswerable":
            continue
        s = r.get("seed_span")
        if isinstance(s, dict) and {"doc_id", "start", "end"} <= s.keys():
            used.add((s["doc_id"], s["start"], s["end"]))
    return used


def write_record(f, record, existing_questions):
    """
    Validate + dedupe (by question text) + write one record.
    Returns "written", "duplicate", or "invalid". Caller is responsible for
    updating any type-specific exclusion sets (used_chunk_ids, seed spans,
    used pairs, etc.) after a successful write.
    """
    if not is_valid_record(record):
        return "invalid"

    q = record["question"].strip().lower()
    if q in existing_questions:
        return "duplicate"

    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()
    existing_questions.add(q)
    return "written" 