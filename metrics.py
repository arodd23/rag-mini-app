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