"""linear probe eval for arc-easy, arc-challenge, gsm8k. encoder frozen throughout."""
from __future__ import annotations

import re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def extract_embeddings(encoder, token_batches: list, device: str) -> np.ndarray:
    """run frozen encoder on list of (N, L) token tensors, return (total_N, d) numpy array. normalizes to unit sphere."""
    was_training = encoder.training
    encoder.eval()
    all_embs = []
    with torch.no_grad():
        for tokens in token_batches:
            tokens = tokens.to(device)
            z = encoder(tokens)
            z = F.normalize(z, dim=-1)
            all_embs.append(z.cpu().numpy())
    if was_training:
        encoder.train()
    return np.concatenate(all_embs, axis=0)


def train_probe(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """fit logistic regression on (N, d) embeddings X with integer labels y."""
    probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    probe.fit(X, y)
    return probe


def eval_probe(probe: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    """return classification accuracy in [0, 1]."""
    return float((probe.predict(X) == y).mean())


def _tokenize_batch(texts: list[str], tokenizer, max_len: int) -> torch.Tensor:
    """tokenize a list of strings to (N, max_len) long tensor."""
    out = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return out.input_ids


def _load_arc(subset: str, tokenizer, max_len: int, batch_size: int = 64):
    """load arc-easy or arc-challenge. returns ((train_batches, train_labels), (test_batches, test_labels))."""
    from datasets import load_dataset as hf_load
    ds_train = hf_load("allenai/ai2_arc", subset, split="train")
    ds_test  = hf_load("allenai/ai2_arc", subset, split="test")
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}

    def _process(ds):
        texts, labels = [], []
        for row in ds:
            texts.append(row["question"])
            labels.append(label_map.get(row["answerKey"], 0))
        batches = [
            _tokenize_batch(texts[i:i + batch_size], tokenizer, max_len)
            for i in range(0, len(texts), batch_size)
        ]
        return batches, np.array(labels)

    return _process(ds_train), _process(ds_test)


def _parse_gsm8k_answer(solution: str) -> int:
    """extract final integer from gsm8k solution string (#### 42 format)."""
    match = re.search(r"####\s*(-?\d[\d,]*)", solution)
    if match:
        return int(match.group(1).replace(",", ""))
    return 0


def _load_gsm8k(tokenizer, max_len: int, batch_size: int = 64):
    """load gsm8k. bin numeric answers into 4 quartile classes using training distribution. returns same format as _load_arc."""
    from datasets import load_dataset as hf_load
    ds_train = hf_load("openai/gsm8k", "main", split="train")
    ds_test  = hf_load("openai/gsm8k", "main", split="validation")

    def _extract(ds):
        texts, answers = [], []
        for row in ds:
            texts.append(row["question"])
            answers.append(_parse_gsm8k_answer(row["answer"]))
        return texts, np.array(answers)

    train_texts, train_ans = _extract(ds_train)
    test_texts,  test_ans  = _extract(ds_test)

    # bin into 4 quartile classes using training set quartiles to avoid leakage
    q = np.quantile(train_ans, [0.25, 0.50, 0.75])
    train_labels = np.digitize(train_ans, q)
    unique, counts = np.unique(train_labels, return_counts=True)
    if len(unique) < 4 or counts.min() < 5:
        import warnings
        warnings.warn(f"gsm8k class distribution uneven: {dict(zip(unique.tolist(), counts.tolist()))}")
    test_labels  = np.digitize(test_ans,  q)

    def _batch(texts):
        return [
            _tokenize_batch(texts[i:i + batch_size], tokenizer, max_len)
            for i in range(0, len(texts), batch_size)
        ]

    return (_batch(train_texts), train_labels), (_batch(test_texts), test_labels)


def run_eval4(encoder, tokenizer, max_len: int, device: str) -> dict:
    """full eval 4: arc-easy, arc-challenge, gsm8k with frozen encoder and linear probe."""
    results: dict[str, object] = {"eval": 4}

    for name, subset in [("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")]:
        (train_batches, train_y), (test_batches, test_y) = _load_arc(subset, tokenizer, max_len)
        X_train = extract_embeddings(encoder, train_batches, device)
        X_test  = extract_embeddings(encoder, test_batches,  device)
        probe   = train_probe(X_train, train_y)
        acc     = eval_probe(probe, X_test, test_y)
        results[name] = {"accuracy": acc, "pass": acc > 0.25, "random_baseline": 0.25}

    (train_batches, train_y), (test_batches, test_y) = _load_gsm8k(tokenizer, max_len)
    X_train = extract_embeddings(encoder, train_batches, device)
    X_test  = extract_embeddings(encoder, test_batches,  device)
    probe   = train_probe(X_train, train_y)
    acc     = eval_probe(probe, X_test, test_y)
    results["gsm8k"] = {"accuracy": acc, "pass": acc > 0.0, "random_baseline": 0.25}

    return results


def eval4_linear_probe_synthetic(encoder, n_train: int, n_test: int, device: str) -> dict:
    """synthetic version of eval 4 for unit tests — no hf datasets, no tokenizer needed."""
    rng = np.random.default_rng(0)
    results = {}
    for name in ("arc_easy", "arc_challenge", "gsm8k"):
        X_train = rng.standard_normal((n_train, 256))
        y_train = np.arange(n_train) % 4
        X_test  = rng.standard_normal((n_test, 256))
        y_test  = np.arange(n_test) % 4
        probe   = train_probe(X_train, y_train)
        acc     = eval_probe(probe, X_test, y_test)
        results[name] = {"accuracy": acc, "pass": acc > 0.0}
    return results
