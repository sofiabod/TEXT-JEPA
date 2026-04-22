import re
from torch.utils.data import Dataset


def segment_paragraphs(text: str, min_chars: int = 50) -> list[str]:
    """split text on double newline, drop segments shorter than min_chars."""
    paras = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paras if len(p.strip()) >= min_chars]


class PG19SegmentDataset(Dataset):
    """loads pg-19 books, concatenates paragraph segments in order, then applies temporal split. max_books limits books loaded for unit tests."""
    def __init__(self, split: str = "train", min_chars: int = 50, max_books: int = None):
        assert split in ("train", "val", "test")
        try:
            from datasets import load_dataset
            raw = load_dataset("pg19", split="train", trust_remote_code=True)
        except Exception:
            raw = []

        all_segments = []
        books = list(raw) if raw else []
        if max_books:
            books = books[:max_books]
        for book in books:
            text = book.get("text", "") if isinstance(book, dict) else ""
            all_segments.extend(segment_paragraphs(text, min_chars=min_chars))

        n = len(all_segments)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.80)
        if split == "train":
            indices = list(range(0, train_end))
        elif split == "val":
            indices = list(range(train_end, val_end))
        else:
            indices = list(range(val_end, n))

        self.segments  = [all_segments[i] for i in indices]
        self.positions = indices

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return {"segment": self.segments[idx], "position": self.positions[idx]}
