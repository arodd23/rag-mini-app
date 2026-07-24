from corpus import ChunkConfig, chunk_fixed, chunk_recursive


def assert_valid(spans, text_len):
    assert spans
    for start, end in spans:
        assert 0 <= start < end <= text_len
    assert spans == sorted(spans)


def test_fixed_overlap_and_coverage():
    text = "word " * 300
    spans = chunk_fixed(text, chunk_size=100, overlap=20)
    assert_valid(spans, len(text))
    assert spans[1][0] == spans[0][1] - 20
    assert spans[0][0] == 0 and spans[-1][1] == len(text)


def test_recursive_overlap_zero_does_not_duplicate_last_sentence():
    text = "One sentence. Two sentence. Three sentence. Four sentence."
    spans = chunk_recursive(text, chunk_size=28, overlap=0)
    assert_valid(spans, len(text))
    assert len(spans) == len(set(spans))


def test_config_validation():
    try:
        ChunkConfig(strategy="fixed", chunk_size=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid overlap to fail")


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\nAll {len(tests)} chunking tests passed.")
