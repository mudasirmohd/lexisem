# Reproducing on Code Ocean (or any offline capsule)

The default Reproducible Run reproduces the **classical** experiment
(baseline vs +lexico-semantic, 5-fold on `movie_reviews`). It is fast and
CPU-friendly. The GPU **LSTM** path (`scripts/run_lstm.py`) is longer and is
kept as a separate, documented step (see below).

## 1. Environment

Tell the capsule to install the Python dependencies. In the **Environment**
tab, add to the post-install / setup:

```
pip install -r requirements.txt
```

(`torch` is only needed for the optional LSTM path; the classical run works
without it.)

## 2. Pre-stage data (avoids the no-internet problem)

A Reproducible Run is usually **sandboxed without internet**, so the runtime
GloVe (~128 MB) and NLTK downloads would fail. Stage them ahead of time.

On any machine **with** internet (e.g. locally, or in the capsule's editable
session which does have network):

```bash
python scripts/prepare_offline_data.py --out data
```

This writes:

```
data/
  embeddings/glove-wiki-gigaword-100.bin    # single-file vectors
  nltk_data/...                             # movie_reviews, wordnet, punkt, ...
```

Upload the contents of `data/` into the capsule's **`data/`** folder. On Code
Ocean that mounts read-only at `/data` during the run.

## 3. Run

The entry point is [`run.sh`](run.sh). Point the capsule's run field at it
(Code Ocean's `code/run`), or just click **Reproducible Run**. It:

- uses `/data/embeddings/glove-wiki-gigaword-100.bin` if present (else downloads),
- sets `NLTK_DATA=/data/nltk_data` if present (else downloads),
- writes `results.json` to `/results`.

`run.sh` also works outside a capsule (`./run.sh`) — it falls back to
`./data` and `./results`, and to runtime downloads if nothing is pre-staged.

## 4. Optional: the LSTM path (GPU)

Not part of the default run (long; needs a GPU for reasonable time). To
reproduce the zero- vs semantic-padding comparison and its locked tables:

```bash
python scripts/run_lstm.py            # both datasets, both readouts, 5-fold
```

Writes per-fold CSVs, summaries and `lstm_stats.json` under
`results_stats_lstm/`. See
[`results_stats_lstm/LOCKED_lstm_results.md`](results_stats_lstm/LOCKED_lstm_results.md)
for the locked results and interpretation.

## 5. What "success" looks like

The classical run prints a results table and writes `/results/results.json`
(SVM + RF, baseline vs lexico_semantic, precision/recall/F1 per class). The
locked, statistically-tested tables that back the README's Results section
live under `results_stats/`, `results_stats_gold/` and `results_stats_lstm/`.
