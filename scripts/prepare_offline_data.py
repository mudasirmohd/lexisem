"""Pre-stage embeddings + NLTK data for an OFFLINE run (e.g. a Code Ocean
capsule whose Reproducible Run has no internet access).

Run this ONCE on a machine WITH internet. It populates a local ``data/``
directory that you then upload to the capsule's ``data/`` folder:

    python scripts/prepare_offline_data.py            # -> ./data
    python scripts/prepare_offline_data.py --out data --pretrained glove-wiki-gigaword-100

Layout produced (matches what run.sh expects):

    data/
      embeddings/<pretrained>.bin       # single-file word2vec-format vectors
      nltk_data/...                     # corpora + models (punkt, wordnet, ...)

Then in the capsule set NLTK_DATA=<.../data/nltk_data> (run.sh does this) and
pass --vectors <.../data/embeddings/<pretrained>.bin>.
"""
from __future__ import annotations
import argparse
import os

import nltk

# Everything the pipeline touches at run time (classical + LSTM + both datasets).
NLTK_RESOURCES = [
    "movie_reviews", "twitter_samples", "opinion_lexicon",
    "wordnet", "omw-1.4", "punkt", "punkt_tab", "stopwords",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output data dir")
    ap.add_argument("--pretrained", default="glove-wiki-gigaword-100",
                    help="gensim-downloader model to stage as a .bin")
    args = ap.parse_args()

    emb_dir = os.path.join(args.out, "embeddings")
    nltk_dir = os.path.join(args.out, "nltk_data")
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(nltk_dir, exist_ok=True)

    print(f"[1/2] NLTK data -> {nltk_dir}")
    for r in NLTK_RESOURCES:
        ok = nltk.download(r, download_dir=nltk_dir, quiet=True)
        print(("  OK  " if ok else "  FAIL"), r)

    print(f"[2/2] embeddings '{args.pretrained}' -> {emb_dir}")
    import gensim.downloader as api
    kv = api.load(args.pretrained)
    out_bin = os.path.join(emb_dir, f"{args.pretrained}.bin")
    kv.save_word2vec_format(out_bin, binary=True)   # single file, .bin loader
    size_mb = os.path.getsize(out_bin) / 1e6
    print(f"  wrote {out_bin}  ({size_mb:.0f} MB, vocab={len(kv):,}, "
          f"dim={kv.vector_size})")

    print(f"\nDone. Upload the contents of '{args.out}/' to the capsule's "
          f"data/ folder, then click Reproducible Run.")


if __name__ == "__main__":
    main()
