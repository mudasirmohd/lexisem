"""Distributional semantic model.

On a local machine you should use *pretrained* embeddings — this matches the
original paper (Google-News Word2Vec / GloVe / FastText) and gives far better
lexicon-expansion quality than corpus-trained vectors.

`load_vectors()` tries, in order:
  1. A local vectors file you point it at (KeyedVectors / .bin / .txt / .vec).
  2. A gensim-downloader model name (auto-downloads the first time).
  3. Fallback: train Word2Vec/FastText on bundled NLTK corpora (no download).

All three return a gensim KeyedVectors object exposing `.most_similar`,
`.key_to_index`, `.vector_size` and `__getitem__`, which is all the rest of
the package needs.
"""
from __future__ import annotations
import os
from typing import Optional

import nltk

DEFAULT_PRETRAINED = "word2vec-google-news-300"  # 1.6 GB, best quality
LIGHTER_PRETRAINED = "glove-wiki-gigaword-100"    # 128 MB, good + fast


def load_vectors(
    local_path: Optional[str] = None,
    pretrained: Optional[str] = LIGHTER_PRETRAINED,
    train_fallback: bool = True,
):
    """Return a gensim KeyedVectors object.

    Parameters
    ----------
    local_path : path to a vectors file on disk (highest priority).
    pretrained : gensim-downloader model name (used if no local_path).
                 e.g. "word2vec-google-news-300", "glove-wiki-gigaword-100",
                 "fasttext-wiki-news-subwords-300".
    train_fallback : if download fails/offline, train on bundled corpora.
    """
    from gensim.models import KeyedVectors

    if local_path and os.path.exists(local_path):
        binary = local_path.endswith(".bin")
        if local_path.endswith((".txt", ".vec", ".bin")):
            return KeyedVectors.load_word2vec_format(local_path, binary=binary)
        return KeyedVectors.load(local_path)

    if pretrained:
        try:
            import gensim.downloader as api
            print(f"[embeddings] loading pretrained '{pretrained}' "
                  f"(first run downloads it) ...")
            return api.load(pretrained)
        except Exception as e:  # offline / blocked
            print(f"[embeddings] pretrained load failed ({e!r}); "
                  f"falling back to local training.")
            if not train_fallback:
                raise

    return _train_local().wv


# ----------------------------------------------------------------------------
# Local-training fallback (self-contained, no downloads beyond NLTK corpora).
# ----------------------------------------------------------------------------
_TRAIN_CORPORA = ["brown", "reuters", "gutenberg", "webtext", "movie_reviews"]


class _SentenceStream:
    def __iter__(self):
        from nltk.corpus import brown, reuters, gutenberg, webtext, movie_reviews
        for src in (brown, reuters, gutenberg, webtext, movie_reviews):
            try:
                for sent in src.sents():
                    toks = [t.lower() for t in sent if t.isalpha()]
                    if len(toks) >= 3:
                        yield toks
            except Exception:
                continue


def _train_local(arch: str = "word2vec", vector_size: int = 100,
                 window: int = 5, min_count: int = 5, epochs: int = 5,
                 seed: int = 42, cache_path: str = "embeddings_lexisem.model"):
    from gensim.models import Word2Vec, FastText
    Model = FastText if arch == "fasttext" else Word2Vec
    if os.path.exists(cache_path):
        return Model.load(cache_path)
    for c in _TRAIN_CORPORA + ["punkt", "punkt_tab"]:
        nltk.download(c, quiet=True)
    common = dict(vector_size=vector_size, window=window,
                  min_count=min_count, workers=os.cpu_count() or 2, seed=seed)
    model = FastText(bucket=2_000_000, sg=1, **common) if arch == "fasttext" \
        else Word2Vec(sg=1, **common)
    stream = _SentenceStream()
    model.build_vocab(stream)
    model.train(stream, total_examples=model.corpus_count, epochs=epochs)
    model.save(cache_path)
    return model
