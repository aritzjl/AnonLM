from anonlm.benchmarking.splits import BenchmarkSplit, build_doc_splits


def test_split_distribution_for_ten_docs() -> None:
    docs = [f"DOC_{i:03d}" for i in range(10)]
    split_map = build_doc_splits(docs)

    assert len(split_map[BenchmarkSplit.DEV]) == 5
    assert len(split_map[BenchmarkSplit.VAL]) == 4
    assert len(split_map[BenchmarkSplit.FINAL]) == 1
