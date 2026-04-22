from dataclasses import dataclass


@dataclass
class ContextWindowCollator:
    k: int = 4

    def __call__(self, segments: list[str]) -> list[dict]:
        """given ordered segments, return sliding windows of k context and 1 target each."""
        windows = []
        for i in range(len(segments) - self.k):
            windows.append({
                "context": segments[i : i + self.k],
                "target": segments[i + self.k],
            })
        return windows
