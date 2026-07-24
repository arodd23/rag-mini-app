"""
build_chunk_index.py

Chunks the corpus once per ChunkConfig and caches the result. Re-run any
time -- it only does real work if the source file or that specific config
has changed since the last build for it (content hash + config stored in
a .meta.json sidecar).

Each distinct ChunkConfig gets its OWN cache file pair, named after
config.name() (e.g. chunk_index__recursive_c1024_o102.jsonl). The default
config (fixed, 500/50) keeps the original unsuffixed filenames for
backward compatibility with the eval-generation scripts and the eval set
they already produced against it.

The cache stores chunk OFFSETS (doc_id, start, end), not chunk text.

Usage:
    python build_chunk_index.py
    python build_chunk_index.py --strategy recursive --chunk-size 1024 --overlap 100
    python build_chunk_index.py --strategy semantic --chunk-size 1024 --similarity-threshold 0.6
    python build_chunk_index.py --rebuild
"""

import argparse
import json
from pathlib import Path

from corpus import load_raw_docs, chunk_document, file_hash, ChunkConfig, DEFAULT_CONFIG

DEFAULT_DATASET_PATH = Path("data/dataset/gov_report_sample_2k.json")
DATA_DIR = Path("data")


def cache_paths(config):
    if config == DEFAULT_CONFIG:
        return DATA_DIR / "chunk_index.jsonl", DATA_DIR / "chunk_index.meta.json"
    name = config.name()
    return DATA_DIR / f"chunk_index__{name}.jsonl", DATA_DIR / f"chunk_index__{name}.meta.json"


def _cache_is_fresh(dataset_path, config, cache_path, meta_path):
    if not (cache_path.exists() and meta_path.exists()):
        return False
    meta = json.loads(meta_path.read_text())
    return (
        meta.get("source_hash") == file_hash(dataset_path)
        and meta.get("source_path") == str(dataset_path)
        and meta.get("config") == config.as_dict()
    )


def build_index(dataset_path=DEFAULT_DATASET_PATH, config=None, embedder=None, force=False):
    """
    Returns (cache_path, meta_path) for this config, building/refreshing
    the cache first if needed.
    """
    config = config or DEFAULT_CONFIG
    dataset_path = Path(dataset_path)
    cache_path, meta_path = cache_paths(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and _cache_is_fresh(dataset_path, config, cache_path, meta_path):
        meta = json.loads(meta_path.read_text())
        print(
            f"Chunk cache '{config.name()}' up to date "
            f"({meta['num_chunks']} chunks from {meta['num_docs']} docs)."
        )
        return cache_path, meta_path

    print(f"Loading documents from {dataset_path}...")
    docs = load_raw_docs(dataset_path)
    print(f"Loaded {len(docs)} documents.")

    if config.strategy == "semantic" and embedder is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model for semantic chunking...")
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunk_id = 0
    n_skipped_short = 0

    print(f"Chunking with config: {config}")
    with cache_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            spans = chunk_document(doc["text"], config=config, embedder=embedder)
            for start, end in spans:
                if end <= start:
                    n_skipped_short += 1
                    continue
                record = {"chunk_id": chunk_id, "doc_id": doc["doc_id"], "start": start, "end": end}
                f.write(json.dumps(record) + "\n")
                chunk_id += 1

    meta = {
        "source_path": str(dataset_path),
        "source_hash": file_hash(dataset_path),
        "config": config.as_dict(),
        "num_docs": len(docs),
        "num_chunks": chunk_id,
        "num_skipped_short": n_skipped_short,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {chunk_id} chunks from {len(docs)} docs -> {cache_path}")
    print(f"Skipped {n_skipped_short} fragments under {config.min_chunk_chars} chars.")
    avg_chunks_per_doc = chunk_id / len(docs) if docs else 0
    print(f"Average chunks per doc: {avg_chunks_per_doc:.1f}")

    return cache_path, meta_path


def load_chunk_index(dataset_path=DEFAULT_DATASET_PATH, config=None, embedder=None):
    """Ensure the cache for this config is built/fresh, then load it."""
    cache_path, _meta_path = build_index(dataset_path=dataset_path, config=config, embedder=embedder, force=False)
    records = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    cfg = ChunkConfig(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        similarity_threshold=args.similarity_threshold,
    )
    build_index(dataset_path=Path(args.dataset), config=cfg, force=args.rebuild)
