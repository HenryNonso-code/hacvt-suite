from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

LABEL_NEG = "neg"
LABEL_NEU = "neu"
LABEL_POS = "pos"


@dataclass
class DomainAdapter:
    token_weights: Dict[str, float]          # per-token delta adjustment
    max_abs_weight: float                    # for clipping
    min_token_count: int                     # tokens below this count ignored
    source: str                              # "supervised" or "weak"
    n_pos: int
    n_neg: int


def _tokenize_basic(text: str) -> List[str]:
    # Keep this consistent with your HACVT tokenizer if you have one.
    # This basic tokenizer is intentionally conservative and explainable.
    out = []
    w = []
    for ch in text.lower():
        if ch.isalnum():
            w.append(ch)
        else:
            if w:
                out.append("".join(w))
                w = []
    if w:
        out.append("".join(w))
    return out


def fit_domain_adapter(
    texts: List[str],
    labels: List[str],
    *,
    min_token_count: int = 5,
    max_abs_weight: float = 1.5,
) -> DomainAdapter:
    """
    Learn per-token weights using smoothed log-odds ratio between pos and neg.
    Neutral items are ignored for learning token polarity drift.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be same length")

    pos_counts = Counter()
    neg_counts = Counter()
    n_pos = 0
    n_neg = 0

    for t, y in zip(texts, labels):
        toks = _tokenize_basic(t)
        if y == LABEL_POS:
            pos_counts.update(toks)
            n_pos += 1
        elif y == LABEL_NEG:
            neg_counts.update(toks)
            n_neg += 1
        else:
            # ignore neutral for token polarity drift
            continue

    vocab = set(pos_counts) | set(neg_counts)
    if n_pos == 0 or n_neg == 0:
        # Not enough signal to learn domain drift
        return DomainAdapter(
            token_weights={},
            max_abs_weight=max_abs_weight,
            min_token_count=min_token_count,
            source="insufficient_signal",
            n_pos=n_pos,
            n_neg=n_neg,
        )

    # Additive smoothing
    alpha = 0.5
    pos_total = sum(pos_counts.values()) + alpha * len(vocab)
    neg_total = sum(neg_counts.values()) + alpha * len(vocab)

    weights: Dict[str, float] = {}
    for tok in vocab:
        c_pos = pos_counts.get(tok, 0)
        c_neg = neg_counts.get(tok, 0)
        if (c_pos + c_neg) < min_token_count:
            continue

        p_pos = (c_pos + alpha) / pos_total
        p_neg = (c_neg + alpha) / neg_total
        w = math.log(p_pos / p_neg)

        # Clip to avoid over-dominance by single tokens
        if w > max_abs_weight:
            w = max_abs_weight
        if w < -max_abs_weight:
            w = -max_abs_weight

        weights[tok] = float(w)

    return DomainAdapter(
        token_weights=weights,
        max_abs_weight=max_abs_weight,
        min_token_count=min_token_count,
        source="supervised",
        n_pos=n_pos,
        n_neg=n_neg,
    )


def fit_domain_adapter_weak(
    texts: List[str],
    preds: List[str],
    *,
    min_token_count: int = 5,
    max_abs_weight: float = 1.5,
) -> DomainAdapter:
    """
    Weak-label version: learn token drift using predicted pos/neg as pseudo-labels.
    Neutral predictions are ignored for learning token drift.
    """
    return fit_domain_adapter(
        texts=texts,
        labels=preds,
        min_token_count=min_token_count,
        max_abs_weight=max_abs_weight,
    )
