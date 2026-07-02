"""End-to-end experiment: baseline vs lexico-semantic features.

Follows the comparison design of Mohd et al. (2022): identical surface features
and classifiers for both conditions; the only difference is whether
lexico-semantic features (from the expanded lexicon) are added. This isolates
their contribution.

Usage examples
--------------
# quick classical run on movie_reviews with a light pretrained GloVe
python scripts/run_experiment.py

# full run with Google-News Word2Vec + the GPU LSTM path
python scripts/run_experiment.py --pretrained word2vec-google-news-300 --lstm

# use your own dataset (CSV with columns: text,label  where label in {0,1})
python scripts/run_experiment.py --csv mydata.csv --lstm
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time
from typing import List, Tuple

import numpy as np
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

from lexisem import load_vectors, load_bing_liu, expand_lexicon, expansion_stats
from lexisem.preprocess import preprocess
from lexisem.features import (
    surface_features, lexico_semantic_features, LEXSEM_KEYS,
)

SEED = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_movie_reviews() -> Tuple[List[str], np.ndarray]:
    nltk.download("movie_reviews", quiet=True)
    from nltk.corpus import movie_reviews
    texts, labels = [], []
    for cat in ("pos", "neg"):
        for fid in movie_reviews.fileids(cat):
            texts.append(movie_reviews.raw(fid))
            labels.append(1 if cat == "pos" else 0)
    return texts, np.array(labels)


def load_csv(path: str) -> Tuple[List[str], np.ndarray]:
    import csv
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, np.array(labels)


# --------------------------------------------------------------------------- #
# Preprocess once, cache to disk
# --------------------------------------------------------------------------- #
def preprocess_all(texts: List[str], cache: str):
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    out = []
    for i, t in enumerate(texts):
        out.append(preprocess(t))          # (lemmas, tags)
        if (i + 1) % 250 == 0:
            print(f"    preprocessed {i+1}/{len(texts)}")
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


def build_feats(tokenised, lex, use_lexsem: bool):
    feats = []
    for lemmas, tags in tokenised:
        f = surface_features(lemmas, tags)
        if use_lexsem:
            f.update(lexico_semantic_features(lemmas, lex))
        feats.append(f)
    return feats


# --------------------------------------------------------------------------- #
# Classical classifiers
#
# The two feature groups get the transform each deserves and are then
# concatenated (FeatureUnion), so neither distorts the other:
#   * sparse surface features (n-grams / negated bigrams / POS) -> TF-IDF
#   * dense lexico-semantic features (the 5 ls_* signals)        -> StandardScaler
# The dense features never flow through TfidfTransformer, and there is no
# hand-tuned weight: standardization puts both groups on comparable footing.
# --------------------------------------------------------------------------- #
class _SurfaceDicts(BaseEstimator, TransformerMixin):
    """Keep only the sparse surface features (drop dense ls_* keys)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        drop = set(LEXSEM_KEYS)
        return [{k: v for k, v in d.items() if k not in drop} for d in X]


class _DenseLexSem(BaseEstimator, TransformerMixin):
    """Extract the fixed ls_* values into a dense (n_samples, 5) matrix."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(
            [[d.get(k, 0.0) for k in LEXSEM_KEYS] for d in X], dtype=float)


def _feature_pipeline(use_lexsem: bool):
    surface = Pipeline([
        ("select", _SurfaceDicts()),
        ("dv", DictVectorizer(sparse=True)),
        ("tfidf", TfidfTransformer()),
    ])
    if not use_lexsem:
        return surface
    lexsem = Pipeline([
        ("select", _DenseLexSem()),
        ("scale", StandardScaler()),
    ])
    return FeatureUnion([("surface", surface), ("lexsem", lexsem)])


def make_clf(name, use_lexsem):
    if name == "svm":
        clf = LinearSVC(C=1.0, random_state=SEED, dual="auto", max_iter=5000)
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    return Pipeline([("feats", _feature_pipeline(use_lexsem)), ("clf", clf)])


def cv_eval(feats, y, name, use_lexsem, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    feats = np.array(feats, dtype=object)
    rows = []
    for tr, te in skf.split(feats, y):
        clf = make_clf(name, use_lexsem)
        clf.fit(list(feats[tr]), y[tr])
        pred = clf.predict(list(feats[te]))
        p, r, f, _ = precision_recall_fscore_support(
            y[te], pred, labels=[1, 0], zero_division=0)
        rows.append([p[0], r[0], f[0], p[1], r[1], f[1]])
    m = np.mean(rows, axis=0)
    return dict(PP=m[0], RP=m[1], F1_P=m[2], PN=m[3], RN=m[4], F1_N=m[5])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV with columns text,label")
    ap.add_argument("--pretrained", default="glove-wiki-gigaword-100",
                    help="gensim-downloader model name, or 'none' to train locally")
    ap.add_argument("--vectors", default=None, help="path to local vectors file")
    ap.add_argument("--sim", type=float, default=0.75)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--lstm", action="store_true", help="also run the LSTM path")
    ap.add_argument("--max-len", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    t0 = time.time()
    tag = "csv" if args.csv else "movie_reviews"

    print("[1/5] Loading embeddings ...")
    pretrained = None if args.pretrained == "none" else args.pretrained
    kv = load_vectors(local_path=args.vectors, pretrained=pretrained)
    print(f"      vocab={len(kv.key_to_index):,}  dim={kv.vector_size}")

    print("[2/5] Loading + expanding Bing Liu lexicon ...")
    base = load_bing_liu()
    exp = expand_lexicon(base, kv, sim_threshold=args.sim, topn=args.topn)
    stats = expansion_stats(base, exp)
    print("      " + "  ".join(f"{k}={v}" for k, v in stats.items()))

    print("[3/5] Loading dataset ...")
    texts, y = load_csv(args.csv) if args.csv else load_movie_reviews()
    print(f"      docs={len(texts)}  pos={int(y.sum())}  neg={int((y==0).sum())}")

    print("[4/5] Preprocessing (cached) + feature extraction ...")
    tokenised = preprocess_all(texts, cache=f"preproc_{tag}.pkl")
    base_feats = build_feats(tokenised, exp, use_lexsem=False)
    ls_feats = build_feats(tokenised, exp, use_lexsem=True)

    print("[5/5] Evaluation ...")
    results = {}
    for clf_name in ("svm", "rf"):
        results[clf_name] = {
            "baseline": cv_eval(base_feats, y, clf_name, use_lexsem=False),
            "lexico_semantic": cv_eval(ls_feats, y, clf_name, use_lexsem=True),
        }

    if args.lstm:
        print("      LSTM path (zero-padding vs semantic-padding) ...")
        from lexisem.deep import train_and_eval
        docs_tokens = [lem for lem, _ in tokenised]
        results["lstm"] = {
            "baseline": train_and_eval(docs_tokens, y, kv, exp,
                                       padding="zero", max_len=args.max_len,
                                       epochs=args.epochs),
            "lexico_semantic": train_and_eval(docs_tokens, y, kv, exp,
                                              padding="semantic", max_len=args.max_len,
                                              epochs=args.epochs),
        }

    out = {"dataset": tag, "embeddings": args.vectors or pretrained or "local",
           "expansion_stats": stats, "results": results,
           "runtime_sec": round(time.time() - t0, 1)}
    with open("results.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n===== RESULTS =====")
    hdr = f"{'model':10s} {'condition':16s} {'PP':>6} {'RP':>6} {'F1-P':>6} {'PN':>6} {'RN':>6} {'F1-N':>6}"
    print(hdr); print("-" * len(hdr))
    for model in results:
        for cond in ("baseline", "lexico_semantic"):
            r = results[model][cond]
            print(f"{model:10s} {cond:16s} "
                  f"{r['PP']:.3f} {r['RP']:.3f} {r['F1_P']:.3f} "
                  f"{r['PN']:.3f} {r['RN']:.3f} {r['F1_N']:.3f}")
    print(f"\nRuntime: {out['runtime_sec']}s   -> results.json")


if __name__ == "__main__":
    main()
