import csv
from torch.utils.data import Dataset


class ROCStoriesDataset(Dataset):
    def __init__(self, csv_path: str, split: str = "train"):
        assert split in ("train", "val", "test")
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n = len(rows)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.80)
        if split == "train":
            self.indices = list(range(0, train_end))
        elif split == "val":
            self.indices = list(range(train_end, val_end))
        else:
            self.indices = list(range(val_end, n))
        self._rows = rows

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self._rows[self.indices[idx]]
        segments = [
            row["sentence1"], row["sentence2"], row["sentence3"],
            row["sentence4"], row["sentence5"],
        ]
        return {"segments": segments, "story_id": row["storyid"]}
