"""
corpus.py

Shared utilities for loading the GovReport corpus and chunking it.

Chunking is now a proper interface with three strategies (Phase 4/5):
    fixed      - the Phase 1 baseline: whitespace-aware fixed-size windows.
    recursive  - paragraph-then-sentence: keep whole paragraphs if they fit,
                 otherwise split into sentences and greedily pack them.
    semantic   - group adjacent sentences while their embedding stays
                 similar to the running chunk, capped at a max size.

All three return the same thing: a list of (start, end) character offsets
into the ORIGINAL doc text. That's what makes everything downstream
(gold_spans, retrieval scoring) strategy-agnostic.

ChunkConfig is a small frozen dataclass that names itself
(e.g. "fixed_c512_o50", "semantic_t60_m1024") -- build_chunk_index.py and
rag_core.py use that name to keep each config's cache files separate, so
switching configs never forces you to recompute one you've already built.

Note on this specific corpus: GovReport doc text appears to have no
paragraph breaks (no blank lines) in the sample we've seen -- so in
practice `recursive` will usually fall straight through to sentence
packing, same as if there were one giant paragraph per doc. That's
expected, not a bug.
"""

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_docs(json_path):
    """
    Load the downloaded GovReport sample JSON.

    Expects a JSON array of records shaped like:
        {"doc_id": 0, "text": "...", "source_id": null, "source_split": "train"}

    Returns a list of {"doc_id": int, "text": str}, skipping empty docs.
    """
    import json

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {json_path}. "
            "Point --dataset (or DEFAULT_DATASET_PATH) at your downloaded gov_report JSON."
        )

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for d in raw:
        text = (d.get("text") or "").strip()
        if not text or "doc_id" not in d:
            continue
        docs.append({"doc_id": d["doc_id"], "text": text})

    return docs


def file_hash(path, block_size=65536):
    """SHA-256 hash of a file's contents, used to detect a stale cache."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Chunk config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "fixed"              # "fixed" | "recursive" | "semantic"
    chunk_size: int = 500                # target chars (fixed/recursive); max chars (semantic)
    overlap: int = 50                    # chars; fixed/recursive only, ignored by semantic
    similarity_threshold: float = 0.6    # semantic only
    min_chunk_chars: int = 200           # all strategies: drop/merge fragments below this

    def __post_init__(self):
        if self.strategy not in {"fixed", "recursive", "semantic"}:
            raise ValueError(f"Unknown chunk strategy: {self.strategy!r}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.min_chunk_chars < 0:
            raise ValueError("min_chunk_chars cannot be negative")
        if self.strategy != "semantic" and not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

    def name(self):
        if self.strategy == "semantic":
            return (
                f"semantic_t{int(round(self.similarity_threshold * 100))}"
                f"_m{self.chunk_size}_min{self.min_chunk_chars}"
            )
        return (
            f"{self.strategy}_c{self.chunk_size}_o{self.overlap}"
            f"_min{self.min_chunk_chars}"
        )

    def as_dict(self):
        return asdict(self)


DEFAULT_CONFIG = ChunkConfig()  # fixed, 500/50 -- the original Phase 1 baseline


# ---------------------------------------------------------------------------
# Sentence / paragraph splitting (regex-based, no heavy NLP dependency)
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")


def split_into_sentences_with_offsets(text):
    """
    Return [(start, end), ...] sentence spans. Heuristic (splits on
    ./!/? followed by whitespace) -- doesn't special-case abbreviations
    like "U.S." or decimals. Good enough for chunk boundaries; not meant
    for anything that needs perfect sentence segmentation.
    """
    if not text:
        return []
    spans = []
    start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def split_into_paragraphs_with_offsets(text):
    """Return [(start, end), ...] paragraph spans, split on blank lines."""
    if not text:
        return []
    spans = []
    start = 0
    for m in _PARAGRAPH_BOUNDARY_RE.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _pack_spans(unit_spans, chunk_size, overlap):
    """Greedily pack ordered spans without dropping text.

    Chunks end before adding a unit that would exceed ``chunk_size`` whenever
    possible. A single sentence longer than the target remains intact because
    splitting sentences would no longer be recursive sentence chunking.
    """
    if not unit_spans:
        return []
    chunks = []
    buffer = []

    def span_len(items):
        return items[-1][1] - items[0][0]

    def trailing_overlap(items):
        if overlap <= 0:
            return []
        kept = []
        for item in reversed(items):
            kept.insert(0, item)
            if kept[-1][1] - kept[0][0] >= overlap:
                break
        return kept

    for unit in unit_spans:
        if buffer and unit[1] - buffer[0][0] > chunk_size:
            chunks.append((buffer[0][0], buffer[-1][1]))
            buffer = trailing_overlap(buffer)
            while buffer and unit[1] - buffer[0][0] > chunk_size:
                buffer.pop(0)
        buffer.append(unit)

    if buffer:
        final = (buffer[0][0], buffer[-1][1])
        if not chunks or final != chunks[-1]:
            chunks.append(final)
    return chunks


def _merge_short_spans(spans, min_chunk_chars, max_chunk_chars=None):
    """Merge short fragments with a neighbor instead of silently dropping them."""
    if not spans or min_chunk_chars <= 0:
        return spans
    merged = []
    for start, end in spans:
        if merged and end - start < min_chunk_chars:
            prev_start, prev_end = merged[-1]
            combined_len = end - prev_start
            if max_chunk_chars is None or combined_len <= max_chunk_chars:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_chunk_chars:
        first = merged.pop(0)
        next_start, next_end = merged[0]
        combined = (first[0], next_end)
        if max_chunk_chars is None or combined[1] - combined[0] <= max_chunk_chars:
            merged[0] = combined
        else:
            merged.insert(0, first)
    return merged


# ---------------------------------------------------------------------------
# Strategy 1: fixed (Phase 1 baseline)
# ---------------------------------------------------------------------------

def chunk_fixed(text, chunk_size=500, overlap=50):
    """Whitespace-aware fixed-size chunking. Does not chunk across documents."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    n = len(text)
    spans = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        if end < n and not text[end].isspace():
            next_space = text.find(" ", end)
            if next_space != -1 and next_space - end < 40:
                end = next_space
        spans.append((start, end))
        if end >= n:
            break
        next_start = max(start + 1, end - overlap)
        start = next_start

    return spans


# ---------------------------------------------------------------------------
# Strategy 2: recursive (paragraph, then sentence)
# ---------------------------------------------------------------------------

def chunk_recursive(text, chunk_size=500, overlap=50, min_chunk_chars=0):
    paragraphs = split_into_paragraphs_with_offsets(text)
    spans = []

    for p_start, p_end in paragraphs:
        if p_end - p_start <= chunk_size:
            spans.append((p_start, p_end))
            continue

        para_text = text[p_start:p_end]
        sent_spans = split_into_sentences_with_offsets(para_text)
        sent_spans = [(p_start + s, p_start + e) for s, e in sent_spans]

        if not sent_spans:
            spans.append((p_start, p_end))
            continue

        spans.extend(_pack_spans(sent_spans, chunk_size, overlap))

    return _merge_short_spans(spans, min_chunk_chars)


# ---------------------------------------------------------------------------
# Strategy 3: semantic (group adjacent sentences by embedding similarity)
# ---------------------------------------------------------------------------

def chunk_semantic(text, embedder, chunk_size=1024, similarity_threshold=0.6, min_chunk_chars=200):
    """
    embedder: a loaded SentenceTransformer (caller-provided so we embed in
    batches across many docs rather than reloading the model per call).
    """
    sent_spans = split_into_sentences_with_offsets(text)
    if not sent_spans:
        return []
    if len(sent_spans) == 1:
        return sent_spans

    sentences = [text[s:e] for s, e in sent_spans]
    embeddings = embedder.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)

    chunks = []
    current_spans = [sent_spans[0]]
    current_emb_sum = embeddings[0].copy()

    for i in range(1, len(sent_spans)):
        centroid = current_emb_sum / len(current_spans)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        sim = float(np.dot(centroid, embeddings[i]))
        candidate_len = sent_spans[i][1] - current_spans[0][0]

        if sim >= similarity_threshold and candidate_len <= chunk_size:
            current_spans.append(sent_spans[i])
            current_emb_sum = current_emb_sum + embeddings[i]
        else:
            chunks.append((current_spans[0][0], current_spans[-1][1]))
            current_spans = [sent_spans[i]]
            current_emb_sum = embeddings[i].copy()

    if current_spans:
        chunks.append((current_spans[0][0], current_spans[-1][1]))

    return _merge_short_spans(chunks, min_chunk_chars, max_chunk_chars=chunk_size)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def chunk_document(text, config=None, embedder=None):
    config = config or DEFAULT_CONFIG

    if config.strategy == "fixed":
        return chunk_fixed(text, config.chunk_size, config.overlap)
    elif config.strategy == "recursive":
        return chunk_recursive(text, config.chunk_size, config.overlap, config.min_chunk_chars)
    elif config.strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an embedder")
        return chunk_semantic(text, embedder, config.chunk_size, config.similarity_threshold, config.min_chunk_chars)
    else:
        raise ValueError(f"unknown chunk strategy: {config.strategy!r}")
