"""Feature engineering.

Surface features (paper 3.4): n-grams, negated bigrams, POS tags.
Lexico-semantic features (paper 3.4.1): polarity hits against the *expanded*
lexicon, given higher weight than surface features.

Also provides `semantic_padding`, the reusable routine that replaces zero
padding with semantically-rich sentiment word vectors for deep-learning
classifiers (paper 3.4.2).
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from .preprocess import NEGATIONS, preprocess
from .lexicon import Lexicon

# Keys of the dense lexico-semantic features, in a fixed order. Consumers that
# need to route these into a separate (standardized, non-TF-IDF) pipeline branch
# can rely on this list.
LEXSEM_KEYS = ["ls_pos", "ls_neg", "ls_pos_ratio", "ls_neg_ratio", "ls_polarity"]


def _ngrams(tokens: List[str], n: int) -> List[str]:
    return ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def surface_features(lemmas: List[str], tags: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    # unigrams + bigrams
    for tok in lemmas:
        feats[f"uni={tok}"] = feats.get(f"uni={tok}", 0.0) + 1.0
    for bg in _ngrams(lemmas, 2):
        feats[f"bi={bg}"] = feats.get(f"bi={bg}", 0.0) + 1.0
    # negated bigrams
    for i in range(len(lemmas) - 1):
        if lemmas[i] in NEGATIONS:
            feats[f"neg={lemmas[i]}_{lemmas[i+1]}"] = 1.0
    # POS tag counts
    for tag in tags:
        feats[f"pos={tag}"] = feats.get(f"pos={tag}", 0.0) + 1.0
    return feats


def lexico_semantic_features(lemmas: List[str], lex: Lexicon) -> Dict[str, float]:
    """Polarity counts/ratios against the (expanded) lexicon.

    These are dense, low-dimensional signals. They are returned on their natural
    scale (no hand-tuned weight): the classifier pipeline is responsible for
    standardizing them and combining them with the sparse surface features, so
    that both groups are on comparable footing without an arbitrary multiplier.
    """
    pos = sum(1 for t in lemmas if t in lex.positive)
    neg = sum(1 for t in lemmas if t in lex.negative)
    total = max(len(lemmas), 1)
    feats = {
        "ls_pos": float(pos),
        "ls_neg": float(neg),
        "ls_pos_ratio": pos / total,
        "ls_neg_ratio": neg / total,
        "ls_polarity": (pos - neg) / total,
    }
    return feats


def extract(
    text: str,
    lex: Lexicon | None = None,
    use_lexsem: bool = False,
) -> Dict[str, float]:
    lemmas, tags = preprocess(text)
    feats = surface_features(lemmas, tags)
    if use_lexsem and lex is not None:
        feats.update(lexico_semantic_features(lemmas, lex))
    return feats


def semantic_padding(
    tokens: List[str],
    kv,
    lex: Lexicon,
    max_len: int = 300,
    vector_size: int | None = None,
    side: str = "post",
) -> np.ndarray:
    """Replace zero padding with semantically-rich sentiment vectors.

    Builds a (max_len, dim) matrix. Real tokens are embedded first; remaining
    slots are filled with vectors of high-similarity sentiment words from the
    expanded lexicon rather than zeros.

    side="post" (default): [real tokens ..., sentiment pads ...].
    side="pre":            [sentiment pads ..., real tokens ...] so the last
        time-step is the last REAL token (pads consumed sequentially first).
    """
    dim = vector_size or kv.vector_size

    def vec(w):
        try:
            return kv[w]
        except KeyError:
            return np.zeros(dim, dtype=np.float32)

    real = [vec(t) for t in tokens[:max_len]]
    pad_rows = []

    if len(real) < max_len:
        # Fill order: sentiment words actually present, then generic lexicon
        # words. Skip out-of-vocabulary filler so padding is genuinely
        # sentiment-bearing rather than zero.
        present = [t for t in tokens if lex.polarity(t) != 0]
        filler = present + sorted(lex.positive) + sorted(lex.negative)
        in_vocab = getattr(kv, "key_to_index", None)
        fi = 0
        need = max_len - len(real)
        while len(pad_rows) < need and fi < len(filler):
            w = filler[fi]
            fi += 1
            if in_vocab is not None and w not in in_vocab:
                continue
            pad_rows.append(vec(w))
        while len(pad_rows) < need:  # safety fallback if lexicon exhausted
            pad_rows.append(np.zeros(dim, dtype=np.float32))

    rows = (pad_rows + real) if side == "pre" else (real + pad_rows)
    return np.vstack(rows[:max_len]).astype(np.float32)
