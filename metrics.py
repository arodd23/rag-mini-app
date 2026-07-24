"""Span-based retrieval metrics that remain valid after rechunking.

A gold span is considered found only when the union of retrieved text covers at
least ``coverage_threshold`` of that span. This avoids counting a one-character
intersection as a complete retrieval hit.

nDCG assigns each gold span to its earliest qualifying retrieved chunk exactly
once, preventing overlapping retrieved chunks from double-counting the same
evidence and guaranteeing 0 <= nDCG <= 1.
"""

import math
from collections import defaultdict

DEFAULT_COVERAGE_THRESHOLD = 0.5


def _validate_span(span):
    required = {"doc_id", "start", "end"}
    if not required.issubset(span):
        raise ValueError(f"Span is missing keys {required - set(span)}: {span}")
    if int(span["end"]) <= int(span["start"]):
        raise ValueError(f"Span end must be greater than start: {span}")


def overlap_length(a, b):
    """Number of overlapping characters, or zero when docs/ranges differ."""
    _validate_span(a)
    _validate_span(b)
    if a["doc_id"] != b["doc_id"]:
        return 0
    return max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))


def spans_overlap(a, b):
    return overlap_length(a, b) > 0


def _merged_intervals(intervals):
    if not intervals:
        return []
    ordered = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def gold_span_coverage(gold_span, retrieved_chunks):
    """Fraction of a gold span covered by the union of retrieved chunks."""
    _validate_span(gold_span)
    intervals = []
    gs, ge = int(gold_span["start"]), int(gold_span["end"])
    for chunk in retrieved_chunks:
        if chunk["doc_id"] != gold_span["doc_id"]:
            continue
        start = max(gs, int(chunk["start"]))
        end = min(ge, int(chunk["end"]))
        if end > start:
            intervals.append((start, end))
    covered = sum(e - s for s, e in _merged_intervals(intervals))
    return covered / (ge - gs)


def _chunk_qualifies_for_gold(chunk, gold, coverage_threshold):
    """Individual-rank match used by MRR/nDCG.

    A single chunk must cover the threshold. Recall uses union coverage, so two
    adjacent smaller chunks can together recover one larger gold span.
    """
    return gold_span_coverage(gold, [chunk]) >= coverage_threshold


def recall_at_k_multi(retrieved_chunks, gold_spans, k, coverage_threshold=DEFAULT_COVERAGE_THRESHOLD):
    if not gold_spans:
        return None
    topk = retrieved_chunks[:k]
    covered = sum(
        gold_span_coverage(gold, topk) >= coverage_threshold
        for gold in gold_spans
    )
    return covered / len(gold_spans)


def mrr_multi(retrieved_chunks, gold_spans, coverage_threshold=DEFAULT_COVERAGE_THRESHOLD):
    if not gold_spans:
        return None
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if any(_chunk_qualifies_for_gold(chunk, gold, coverage_threshold) for gold in gold_spans):
            return 1.0 / rank
    return 0.0


def ndcg_at_k_multi(retrieved_chunks, gold_spans, k, coverage_threshold=DEFAULT_COVERAGE_THRESHOLD):
    """Binary nDCG with one relevance credit per gold span and no duplicates."""
    if not gold_spans:
        return None

    unmatched = set(range(len(gold_spans)))
    relevance = []
    for chunk in retrieved_chunks[:k]:
        matched_index = next(
            (
                i for i in sorted(unmatched)
                if _chunk_qualifies_for_gold(chunk, gold_spans[i], coverage_threshold)
            ),
            None,
        )
        if matched_index is None:
            relevance.append(0.0)
        else:
            relevance.append(1.0)
            unmatched.remove(matched_index)

    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, start=1))
    ideal_hits = min(len(gold_spans), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    score = dcg / idcg if idcg else 0.0
    return min(1.0, max(0.0, score))
