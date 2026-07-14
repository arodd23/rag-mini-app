"""
build_chunk_index.py

Chunks the corpus once and caches the result. Re-run any time -- it only
does real work if the source file, chunk size, overlap, or min-chunk-length
have changed since the last build (checked via a content hash + params
stored in the .meta.json sidecar).

The cache stores chunk OFFSETS (doc_id, start, end), not chunk text, so it
stays small. gen_eval*.py slice the actual text out of the source docs on
demand, and store those same offsets as gold_spans -- durable ground truth
that survives rechunking.

Usage:
    python build_chunk_index.py
    python build_chunk_index.py --rebuild        # force a rebuild
    python build_chunk_index.py --dataset path/to/your.json
"""

import argparse
import json
from pathlib import Path

from corpus import load_raw_docs, chunk_document, file_hash

DEFAULT_DATASET_PATH = Path("data/dataset/gov_report_sample_2k.json")
CACHE_PATH = Path("data/cache/chunk_index.jsonl")
CACHE_META_PATH = Path("data/cache/chunk_index.meta.json")

CHUNK_SIZE = 500
OVERLAP = 50
MIN_CHUNK_CHARS = 200  # drop trailing fragments shorter than this


def _cache_is_fresh(dataset_path):
    if not (CACHE_PATH.exists() and CACHE_META_PATH.exists()):
        return False

    meta = json.loads(CACHE_META_PATH.read_text())
    return (
        meta.get("source_hash") == file_hash(dataset_path)
        and meta.get("source_path") == str(dataset_path)
        and meta.get("chunk_size") == CHUNK_SIZE
        and meta.get("overlap") == OVERLAP
        and meta.get("min_chunk_chars") == MIN_CHUNK_CHARS
    )


def build_index(dataset_path=DEFAULT_DATASET_PATH, force=False):
    dataset_path = Path(dataset_path)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force and _cache_is_fresh(dataset_path):
        meta = json.loads(CACHE_META_PATH.read_text())
        print(
            f"Chunk cache is up to date "
            f"({meta['num_chunks']} chunks from {meta['num_docs']} docs). "
            "Nothing to do. Use --rebuild to force."
        )
        return

    print(f"Loading documents from {dataset_path}...")
    docs = load_raw_docs(dataset_path)
    print(f"Loaded {len(docs)} documents.")

    chunk_id = 0
    n_skipped_short = 0

    print(f"Chunking (size={CHUNK_SIZE}, overlap={OVERLAP})...")
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            spans = chunk_document(doc["text"], CHUNK_SIZE, OVERLAP)
            for start, end in spans:
                if end - start < MIN_CHUNK_CHARS:
                    n_skipped_short += 1
                    continue
                record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc["doc_id"],
                    "start": start,
                    "end": end,
                }
                f.write(json.dumps(record) + "\n")
                chunk_id += 1

    meta = {
        "source_path": str(dataset_path),
        "source_hash": file_hash(dataset_path),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "num_docs": len(docs),
        "num_chunks": chunk_id,
        "num_skipped_short": n_skipped_short,
    }
    CACHE_META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {chunk_id} chunks from {len(docs)} docs -> {CACHE_PATH}")
    print(f"Skipped {n_skipped_short} fragments under {MIN_CHUNK_CHARS} chars.")
    avg_chunks_per_doc = chunk_id / len(docs) if docs else 0
    print(f"Average chunks per doc: {avg_chunks_per_doc:.1f}")


def load_chunk_index(dataset_path=DEFAULT_DATASET_PATH):
    """Ensure the cache is built/fresh, then load it into memory as a list of dicts."""
    build_index(dataset_path=dataset_path, force=False)
    records = []
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    build_index(dataset_path=Path(args.dataset), force=args.rebuild)