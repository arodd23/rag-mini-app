import math

from metrics import (
    gold_span_coverage,
    mrr_multi,
    ndcg_at_k_multi,
    recall_at_k_multi,
    spans_overlap,
)


def span(doc_id, start, end):
    return {"doc_id": doc_id, "start": start, "end": end}


chunk = span


def approx(a, b, eps=1e-6):
    return abs(a - b) < eps


def test_spans_overlap():
    assert spans_overlap(span(0, 0, 10), span(0, 5, 15))
    assert not spans_overlap(span(0, 0, 10), span(0, 10, 20))
    assert not spans_overlap(span(0, 0, 10), span(1, 0, 10))


def test_tiny_overlap_is_not_full_hit():
    gold = [span(0, 100, 200)]
    retrieved = [chunk(0, 199, 250)]
    assert recall_at_k_multi(retrieved, gold, 5) == 0.0
    assert mrr_multi(retrieved, gold) == 0.0


def test_union_coverage_for_smaller_chunks():
    gold = [span(0, 100, 200)]
    retrieved = [chunk(0, 100, 135), chunk(0, 135, 170)]
    assert approx(gold_span_coverage(gold[0], retrieved), 0.70)
    assert recall_at_k_multi(retrieved, gold, 5) == 1.0


def test_gold_at_rank_1():
    gold = [span(0, 100, 200)]
    retrieved = [chunk(0, 100, 200), chunk(0, 500, 600)]
    assert recall_at_k_multi(retrieved, gold, 5) == 1.0
    assert mrr_multi(retrieved, gold) == 1.0
    assert ndcg_at_k_multi(retrieved, gold, 5) == 1.0


def test_gold_at_rank_2():
    gold = [span(0, 100, 200)]
    retrieved = [chunk(1, 0, 100), chunk(0, 100, 200)]
    assert mrr_multi(retrieved, gold) == 0.5
    assert approx(ndcg_at_k_multi(retrieved, gold, 5), 1.0 / math.log2(3))


def test_duplicate_chunks_do_not_inflate_ndcg():
    gold = [span(0, 100, 200)]
    retrieved = [chunk(0, 100, 200), chunk(0, 100, 200), chunk(0, 100, 200)]
    score = ndcg_at_k_multi(retrieved, gold, 5)
    assert score == 1.0
    assert 0.0 <= score <= 1.0


def test_multihop_partial_recall():
    gold = [span(0, 100, 200), span(0, 500, 600)]
    retrieved = [chunk(0, 100, 200), chunk(1, 0, 50)]
    assert recall_at_k_multi(retrieved, gold, 5) == 0.5


def test_unanswerable_returns_none():
    assert recall_at_k_multi([chunk(0, 0, 10)], [], 5) is None
    assert mrr_multi([chunk(0, 0, 10)], []) is None
    assert ndcg_at_k_multi([chunk(0, 0, 10)], [], 5) is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
