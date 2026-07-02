"""LSTM path: zero-padding vs semantic-padding, k-fold with significance.

Reproduces the paper's core deep-learning claim (section 3.4.2) under the SAME
evaluation protocol as the locked classical results:

  * 5-fold StratifiedKFold(shuffle=True, random_state=42)
  * macro-F1 = mean(F1_pos, F1_neg) per fold
  * paired t-test + Wilcoxon on the 5 per-fold deltas (semantic - zero)
  * same fold split / weight init / minibatch order / dropout masks for both
    conditions, so the only difference is the padding

Datasets (labels reconstructed deterministically; the source CSVs are cached
only as tokens in preproc_*.pkl):
  * movie_reviews  — long-form; NLTK order pos(1000)+neg(1000)   -> null control
  * twitter        — short/informal; NLTK twitter_samples order
                     pos(5000)+neg(5000), emoticon distant labels -> effect regime

Usage:
  python scripts/run_lstm.py                       # both datasets, GloVe-100
  python scripts/run_lstm.py --datasets twitter    # one dataset
  python scripts/run_lstm.py --epochs 6 --max-len 300
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from lexisem import load_vectors, load_bing_liu, expand_lexicon, expansion_stats
from lexisem.deep import kfold_compare

OUTDIR = "results_stats_lstm"
SEED = 42

# dataset -> (preproc cache, label vector builder)
DATASETS = {
    "movie": ("preproc_movie_reviews.pkl",
              lambda: np.array([1] * 1000 + [0] * 1000)),
    "twitter": ("preproc_csv.pkl",
                lambda: np.array([1] * 5000 + [0] * 5000)),
}


def load_tokens(cache):
    with open(cache, "rb") as fh:
        tok = pickle.load(fh)
    return [lem for lem, _ in tok]


def paired_stats(zero, sem, zero_col, sem_col):
    zero = np.asarray(zero, float)
    sem = np.asarray(sem, float)
    delta = sem - zero
    t_stat, t_p = ttest_rel(sem, zero)
    # Wilcoxon needs >0 nonzero diffs; guard the degenerate case.
    try:
        w_stat, w_p = wilcoxon(sem, zero)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "mean_zero": float(zero.mean()), "std_zero": float(zero.std(ddof=1)),
        "mean_semantic": float(sem.mean()), "std_semantic": float(sem.std(ddof=1)),
        "mean_delta": float(delta.mean()),
        "n_positive": int((delta > 0).sum()), "n_folds": len(delta),
        "paired_t_stat": float(t_stat), "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "n_collapsed_zero": int(sum(zero_col)),
        "n_collapsed_semantic": int(sum(sem_col)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="both",
                    choices=["both", "movie", "twitter"])
    ap.add_argument("--pretrained", default="glove-wiki-gigaword-100")
    ap.add_argument("--vectors", default=None)
    ap.add_argument("--sim", type=float, default=0.75)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--readouts", default="meanpool,prelast",
                    help="comma list of readouts: meanpool, prelast")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    which = ["movie", "twitter"] if args.datasets == "both" else [args.datasets]

    print("[1/3] Loading embeddings ...")
    pretrained = None if args.pretrained == "none" else args.pretrained
    kv = load_vectors(local_path=args.vectors, pretrained=pretrained)
    emb_name = args.vectors or pretrained or "local"
    print(f"      {emb_name}  vocab={len(kv.key_to_index):,}  dim={kv.vector_size}")

    print("[2/3] Loading + expanding Bing Liu lexicon ...")
    base = load_bing_liu()
    lex = expand_lexicon(base, kv, sim_threshold=args.sim, topn=args.topn)
    stats = expansion_stats(base, lex)
    print("      " + "  ".join(f"{k}={v}" for k, v in stats.items()))

    readouts = [r.strip() for r in args.readouts.split(",") if r.strip()]
    print(f"[3/3] LSTM k-fold — readouts={readouts} ...")

    # cache preprocessed tokens/labels once (reused across readouts)
    data = {}
    for name in which:
        cache, ylab = DATASETS[name]
        docs = load_tokens(cache)
        y = ylab()
        assert len(docs) == len(y), f"{name}: docs {len(docs)} != labels {len(y)}"
        data[name] = (docs, y)

    all_cells = {}
    for readout in readouts:
        print(f"\n########## readout = {readout} ##########")
        per_fold_rows, summary_rows, cells = [], [], {}
        for name in which:
            docs, y = data[name]
            print(f"  == {name}  (N={len(docs)}) ==")
            res = kfold_compare(docs, y, kv, lex, max_len=args.max_len,
                                epochs=args.epochs, lr=args.lr,
                                n_splits=args.folds, seed=SEED, readout=readout)
            st = paired_stats(res["zero_macro_f1"], res["semantic_macro_f1"],
                              res["zero_collapsed"], res["semantic_collapsed"])
            cells[name] = {"folds": res, "stats": st}
            for f, (z, s, zc, sc) in enumerate(zip(res["zero_macro_f1"],
                                                   res["semantic_macro_f1"],
                                                   res["zero_collapsed"],
                                                   res["semantic_collapsed"])):
                per_fold_rows.append((name, f, z, s, s - z, zc, sc))
            summary_rows.append((name, st))
            warn = ""
            if st["n_collapsed_zero"] or st["n_collapsed_semantic"]:
                warn = (f"  [!! collapses remain: zero={st['n_collapsed_zero']} "
                        f"semantic={st['n_collapsed_semantic']}]")
            print(f"     mean Δ={st['mean_delta']:+.4f}  "
                  f"({st['n_positive']}/{st['n_folds']} folds +)  "
                  f"t_p={st['paired_t_p']:.4f}  W_p={st['wilcoxon_p']:.4f}{warn}")

        # ---- write per-readout artifacts ----
        pf = os.path.join(OUTDIR, f"lstm_{readout}_per_fold.csv")
        with open(pf, "w") as fh:
            fh.write("dataset,fold,zero_macro_f1,semantic_macro_f1,delta,"
                     "zero_collapsed,semantic_collapsed\n")
            for name, f, z, s, d, zc, sc in per_fold_rows:
                fh.write(f"{name},{f},{z:.6f},{s:.6f},{d:.6f},"
                         f"{int(zc)},{int(sc)}\n")

        sf = os.path.join(OUTDIR, f"lstm_{readout}_summary.csv")
        with open(sf, "w") as fh:
            fh.write("dataset,mean_zero,std_zero,mean_semantic,std_semantic,"
                     "mean_delta,n_positive,paired_t_stat,paired_t_p,wilcoxon_p,"
                     "n_collapsed_zero,n_collapsed_semantic\n")
            for name, st in summary_rows:
                fh.write(f"{name},{st['mean_zero']:.6f},{st['std_zero']:.6f},"
                         f"{st['mean_semantic']:.6f},{st['std_semantic']:.6f},"
                         f"{st['mean_delta']:.6f},{st['n_positive']},"
                         f"{st['paired_t_stat']:.4f},{st['paired_t_p']:.5f},"
                         f"{st['wilcoxon_p']:.5f},{st['n_collapsed_zero']},"
                         f"{st['n_collapsed_semantic']}\n")
        all_cells[readout] = cells

    out = {
        "status": "LSTM results (zero vs semantic padding)",
        "protocol": (f"{args.folds}-fold StratifiedKFold(shuffle=True, "
                     f"random_state={SEED}); macro-F1=mean(F1_pos,F1_neg); "
                     f"paired t + Wilcoxon on per-fold deltas; identical fold "
                     f"split/init/minibatch order/dropout per condition; "
                     f"epochs={args.epochs}, max_len={args.max_len}, "
                     f"lr={args.lr}, grad_clip=5.0, collapse-retry=1"),
        "readouts": {
            "meanpool": ("post-pad; masked mean-pool (zero over real tokens, "
                         "semantic over all slots)"),
            "prelast": ("pre-pad (real tokens last); last-time-step readout; "
                        "semantic pads consumed sequentially before real tokens"),
        },
        "embeddings": emb_name,
        "expansion_stats": stats,
        "labels_note": ("labels reconstructed deterministically: movie=NLTK "
                        "movie_reviews pos(1000)+neg(1000); twitter=NLTK "
                        "twitter_samples pos(5000)+neg(5000), emoticon distant"),
        "cells_by_readout": all_cells,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUTDIR, "lstm_stats.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nWrote {OUTDIR}/  (runtime {out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
