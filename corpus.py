"""
corpus.py

Shared utilities for loading the GovReport corpus and chunking it.
Used by both build_chunk_index.py (one-time/cached chunking) and
gen_eval.py (eval set generation).
"""

import json
import hashlib
from pathlib import Path


def load_raw_docs(json_path):
    """
    Load the downloaded GovReport sample JSON.

    Expects a JSON array of records shaped like:
        {"doc_id": 0, "text": "...", "source_id": null, "source_split": "train"}

    Returns a list of {"doc_id": int, "text": str}, skipping empty docs.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {json_path}. "
            "Point DATASET_PATH (or --dataset) at your downloaded gov_report JSON."
        )

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for d in raw:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        if "doc_id" not in d:
            continue
        docs.append({"doc_id": d["doc_id"], "text": text})

    return docs


def file_hash(path, block_size=65536):
    """SHA-256 hash of a file's contents, used to detect when the chunk cache is stale."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def chunk_document(text, chunk_size=500, overlap=50):
    """
    Whitespace-aware fixed-size chunking (Phase 1 baseline: 500 chars, 50 overlap).

    Returns a list of (start, end) character offsets into `text`. Chunk
    boundaries are snapped forward to the next space when they'd otherwise
    land mid-word, so chunks don't start/end on a fragment of a word.

    Does NOT chunk across documents -- call once per document.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    n = len(text)
    spans = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)

        # If we're cutting mid-word (not at end of doc, next char isn't
        # whitespace), push the boundary to the next space -- but only if
        # that space is nearby, so one long token can't blow up chunk size.
        if end < n and not text[end].isspace():
            next_space = text.find(" ", end)
            if next_space != -1 and next_space - end < 40:
                end = next_space

        spans.append((start, end))

        if end >= n:
            break
        start += step

    return spans