# lexisem — Lexico-Semantic Features for Sentiment Analysis

Reference implementation of the method in:

> Mohd, Javeed, Nowsheena, Wani, Khanday (2022),
> *Sentiment analysis using lexico-semantic features*,
> **Journal of Information Science**. DOI: 10.1177/01655515221124016

The pipeline (i) expands a sentiment lexicon using a distributional semantic
model plus WordNet, (ii) extracts **lexico-semantic features** from text under
lexicon supervision, and (iii) classifies sentiment. For the deep-learning
path it replaces zero padding with **semantic padding** (sentiment-bearing
word vectors) for an LSTM classifier.

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
pip install -e .                                        # editable install
python scripts/download_nltk.py                         # one-time NLTK data
```

For the GPU LSTM path, install the CUDA build of PyTorch that matches your
driver (see https://pytorch.org), e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 2. Run

Classical models (SVM + Random Forest) on the bundled `movie_reviews` set,
with a light pretrained GloVe (auto-downloads ~128 MB the first time):

```bash
python scripts/run_experiment.py
```

Full run with Google-News Word2Vec (best quality, ~1.6 GB download) **and**
the GPU LSTM zero-vs-semantic-padding comparison:

```bash
python scripts/run_experiment.py --pretrained word2vec-google-news-300 --lstm
```

Your own data (CSV with columns `text,label`, label ∈ {0,1}):

```bash
python scripts/run_experiment.py --csv mydata.csv --lstm
```

Use a local vectors file instead of downloading:

```bash
python scripts/run_experiment.py --vectors /path/to/GoogleNews-vectors-negative300.bin
```

Results print to the console and are written to `results.json`.
Preprocessing is cached (`preproc_*.pkl`) so re-runs are fast.

For a one-click / offline reproducible run (e.g. Code Ocean), use `run.sh`
and pre-stage data with `scripts/prepare_offline_data.py` — see
[`REPRODUCE.md`](REPRODUCE.md).

### Useful flags
| flag | meaning | default |
|------|---------|---------|
| `--pretrained` | gensim model name, or `none` to train embeddings locally | `glove-wiki-gigaword-100` |
| `--vectors` | path to a local `.bin/.txt/.vec` vectors file | — |
| `--sim` | similarity threshold for lexicon expansion | `0.75` |
| `--topn` | neighbours per word during expansion | `10` |
| `--lstm` | also run the LSTM path (needs PyTorch) | off |
| `--max-len` | sequence length for the LSTM | `300` |
| `--epochs` | LSTM training epochs | `6` |
| `--csv` | dataset CSV (`text,label`) | movie_reviews |

## 3. Package layout

```
lexisem/
  preprocess.py   clean / tokenise / lemmatise / POS  (paper §3.1)
  embeddings.py   pretrained loader + local-training fallback  (paper §3.2)
  lexicon.py      Bing Liu / AFINN + Algorithm 1 expansion  (paper §3.2)
  features.py     surface + lexico-semantic features; semantic padding (§3.4)
  deep.py         4-layer LSTM, zero vs semantic padding  (paper §3.4.2)
scripts/
  run_experiment.py   end-to-end baseline vs lexico-semantic
  download_nltk.py     one-time resource download
```

## 4. Notes on faithfulness

- The method is implemented faithfully; **absolute scores depend on your
  embeddings, lexicon version and dataset** and will differ from the 2022
  paper's tables, which used different (some now-unavailable) datasets.
- The default base lexicon is Bing Liu's opinion lexicon (loads at 6,789
  entries, matching the paper). AFINN is also available via `load_afinn()`.
- Swap in any gensim `KeyedVectors`-compatible embedding without code changes.

## 5. Results

All results use 5-fold `StratifiedKFold(shuffle=True, random_state=42)`,
macro-F1 = mean(F1_pos, F1_neg), with paired t-test + Wilcoxon on per-fold
deltas. Locked tables (numbers, per-fold CSVs, provenance) live under
`results_stats*/`.

**Classical — baseline vs +lexico-semantic features**
([`results_stats/LOCKED_classical_results.md`](results_stats/LOCKED_classical_results.md)):
the effect is **negligible on long-form** text (movie_reviews), **small on
short/informal heuristic-labeled** text (twitter_samples), and **largest on
short/informal GOLD** text (Sentiment140, Δ≈+0.03–0.06) — significant when
pooled but per-cell underpowered (small n, high fold variance). Adding dense
features naively (through TF-IDF + a hard weight) harms the SVM; a principled
pipeline (standardize dense features, isolate from TF-IDF) removes the harm and
yields the short-text-concentrated positive effect.

**Deep learning — zero- vs semantic-padding LSTM**
([`results_stats_lstm/LOCKED_lstm_results.md`](results_stats_lstm/LOCKED_lstm_results.md),
reproduce with `python scripts/run_lstm.py`): under a **fair** evaluation
semantic padding shows **no genuine gain**. Tested across two readout families
(masked mean-pool and pre-pad/last-step), the effect is null on short text
(Δ≈+0.003, n.s.) and near-null/negative elsewhere. A naive last-time-step
readout appears to give a large gain (≈+0.20) only because it collapses the
zero-padding baseline to a single class (macro-F1 0.333) — a readout artifact,
not a representational improvement.

> **On faithfulness.** The method is implemented faithfully; absolute scores
> depend on embeddings, lexicon version and dataset, and differ from the 2022
> paper's tables (which used some now-unavailable datasets). These locked
> tables report what a rigorous, principled re-evaluation actually yields.

## 6. License

MIT (see `LICENSE`).
