"""Retrieval metrics for single-gold and multi-gold RAG evaluation items."""

import math
from collections.abc import Iterable


def _unique_ids(ids: Iterable[int]) -> list[int]:
    """Return IDs in original order with duplicates removed."""
    return list(dict.fromkeys(int(item_id) for item_id in ids))


def recall_at_k_multi(retrieved_ids, gold_ids, k=5):
    """
    Fraction of unique gold chunks retrieved in the top-k results.

    Returns None for records with no gold chunks, such as unanswerable
    questions, because retrieval recall is not applicable to those rows.
    """
    gold = set(_unique_ids(gold_ids))
    if not gold:
        return None

    retrieved_top_k = set(_unique_ids(retrieved_ids)[:k])
    return len(gold & retrieved_top_k) / len(gold)


def mrr_multi(retrieved_ids, gold_ids):
    """
    Reciprocal rank of the first retrieved gold chunk.

    Returns None when the record has no gold chunks.
    """
    gold = set(_unique_ids(gold_ids))
    if not gold:
        return None

    for rank, chunk_id in enumerate(_unique_ids(retrieved_ids), start=1):
        if chunk_id in gold:
            return 1.0 / rank

    return 0.0


def ndcg_at_k_multi(retrieved_ids, gold_ids, k=5):
    """
    Binary-relevance nDCG@k for one or more gold chunks.

    The score is normalized by the ideal placement of up to k unique gold
    chunks. Returns None when the record has no gold chunks.
    """
    gold = set(_unique_ids(gold_ids))
    if not gold:
        return None

    ranked_ids = _unique_ids(retrieved_ids)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids, start=1)
        if chunk_id in gold
    )

    ideal_hits = min(len(gold), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


# Backward-compatible single-gold wrappers.
def recall_at_k(retrieved_ids, gold_id, k=5):
    return recall_at_k_multi(retrieved_ids, [gold_id], k=k)


def mrr(retrieved_ids, gold_id):
    return mrr_multi(retrieved_ids, [gold_id])


def ndcg_at_k(retrieved_ids, gold_id, k=5):
    return ndcg_at_k_multi(retrieved_ids, [gold_id], k=k)