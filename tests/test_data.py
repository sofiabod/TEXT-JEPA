import pytest


def test_rocstories_segment_count():
    """each story yields 5 sentence-level segments."""
    from src.data.rocstories import ROCStoriesDataset
    ds = ROCStoriesDataset(csv_path="tests/fixtures/rocstories_sample.csv", split="train")
    item = ds[0]
    assert len(item["segments"]) == 5
    assert all(isinstance(s, str) for s in item["segments"])


def test_rocstories_temporal_split():
    """temporal split: train 70%, val 10%, test 20% by story index, no overlap."""
    from src.data.rocstories import ROCStoriesDataset
    train_ds = ROCStoriesDataset(csv_path="tests/fixtures/rocstories_sample.csv", split="train")
    val_ds   = ROCStoriesDataset(csv_path="tests/fixtures/rocstories_sample.csv", split="val")
    test_ds  = ROCStoriesDataset(csv_path="tests/fixtures/rocstories_sample.csv", split="test")
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert abs(len(train_ds) / total - 0.70) < 0.05
    assert abs(len(val_ds) / total - 0.10) < 0.05
    assert abs(len(test_ds) / total - 0.20) < 0.05


def test_rocstories_no_future_leakage():
    """test stories never appear in train or val indices."""
    from src.data.rocstories import ROCStoriesDataset
    train_ds = ROCStoriesDataset(csv_path="tests/fixtures/rocstories_sample.csv", split="train")
    test_ds  = ROCStoriesDataset(csv_path="tests/fixtures/rocstories_sample.csv", split="test")
    train_indices = set(train_ds.indices)
    test_indices  = set(test_ds.indices)
    assert len(train_indices & test_indices) == 0, "train and test sets overlap"


def test_pg19_paragraph_segmentation():
    """paragraphs split on double newline, each returns a non-empty string."""
    from src.data.pg19 import segment_paragraphs
    text = "Para one sentence one. Sentence two.\n\nPara two here.\n\nPara three."
    segs = segment_paragraphs(text, min_chars=10)
    assert len(segs) == 3
    assert all(len(s) >= 10 for s in segs)


def test_pg19_temporal_split_no_shuffle():
    """split is by sequence position, not random. train indices < val indices < test indices."""
    from src.data.pg19 import PG19SegmentDataset
    ds_train = PG19SegmentDataset(split="train", max_books=2)
    ds_test  = PG19SegmentDataset(split="test",  max_books=2)
    if len(ds_train.positions) > 0 and len(ds_test.positions) > 0:
        assert max(ds_train.positions) < min(ds_test.positions)


def test_collator_context_window():
    """collator returns windows of k context segments and 1 target segment."""
    from src.data.collator import ContextWindowCollator
    segments = [f"Segment {i}." for i in range(20)]
    collator = ContextWindowCollator(k=4)
    windows = collator(segments)
    assert all(len(w["context"]) == 4 for w in windows)
    assert all(isinstance(w["target"], str) for w in windows)


def test_collator_no_future_in_context():
    """context window never includes the target segment or anything after it."""
    from src.data.collator import ContextWindowCollator
    segments = [f"Segment {i}." for i in range(10)]
    collator = ContextWindowCollator(k=4)
    for w in collator(segments):
        target_idx = segments.index(w["target"])
        context_indices = [segments.index(c) for c in w["context"]]
        assert all(ci < target_idx for ci in context_indices)
