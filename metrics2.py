import math


def recall_at_k(retrieved_ids, gold_id, k=5):
    """
    Returns 1 if the gold chunk appears in the top-k retrieved chunks.
    Otherwise returns 0.
    """
    return 1 if gold_id in retrieved_ids[:k] else 0


def mrr(retrieved_ids, gold_id):
    """
    Mean Reciprocal Rank for one question.
    If gold chunk is ranked 1st, score = 1.
    If ranked 2nd, score = 1/2.
    If not found, score = 0.
    """
    for index, chunk_id in enumerate(retrieved_ids):
        if chunk_id == gold_id:
            return 1 / (index + 1)

    return 0


def ndcg_at_k(retrieved_ids, gold_id, k=5):
    """
    nDCG@k for one correct chunk.
    Rewards the gold chunk appearing higher in the ranking.
    """
    for index, chunk_id in enumerate(retrieved_ids[:k]):
        if chunk_id == gold_id:
            return 1 / math.log2(index + 2)

    return 0


# ================================================================
# MULTI-GOLD VERSIONS
# ================================================================

def recall_at_k_multi(retrieved_ids, gold_ids, k=5):
    """
    Fraction of gold chunks found in the top-k retrieved chunks.
    Returns None if there are no gold chunks to find (e.g. an
    unanswerable question) -- there is nothing to score.
    """
    if not gold_ids:
        return None

    gold_set = set(gold_ids)
    hits = len(gold_set & set(retrieved_ids[:k]))

    return hits / len(gold_set)


def mrr_multi(retrieved_ids, gold_ids):
    """
    Reciprocal rank of the first retrieved chunk that is in the
    gold set. Returns None if there are no gold chunks.
    """
    if not gold_ids:
        return None

    gold_set = set(gold_ids)

    for index, chunk_id in enumerate(retrieved_ids):
        if chunk_id in gold_set:
            return 1 / (index + 1)

    return 0


def ndcg_at_k_multi(retrieved_ids, gold_ids, k=5):
    """
    nDCG@k against a set of gold chunks, normalized by the ideal
    ranking (all gold chunks, up to k of them, at the top).
    Returns None if there are no gold chunks.

    For a single gold id this reduces to the same score as
    ndcg_at_k() above.
    """
    if not gold_ids:
        return None

    gold_set = set(gold_ids)

    dcg = sum(
        1 / math.log2(index + 2)
        for index, chunk_id in enumerate(retrieved_ids[:k])
        if chunk_id in gold_set
    )

    ideal_hits = min(len(gold_set), k)

    idcg = sum(
        1 / math.log2(index + 2)
        for index in range(ideal_hits)
    )

    if idcg == 0:
        return 0

    return dcg / idcg