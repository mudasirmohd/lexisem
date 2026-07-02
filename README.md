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

## 4. Notes on faithfulness and results

`lexisem` is a clean-room reimplementation of the lexico-semantic
method. It faithfully implements the method's components (lexicon
expansion, lexico-semantic feature extraction, semantic padding). As
with any reimplementation, absolute scores depend on the choice of
embeddings, lexicon version, and evaluation datasets, so they are not
expected to match any specific prior run. The results reported here
characterize the method's behaviour under a rigorous, principled
evaluation on public benchmark datasets, rather than reproducing a
particular set of published numbers.

Two findings are worth stating plainly, because the software is
designed to surface them honestly:

- **Feature combination matters more than feature presence.** When the
  dense lexico-semantic features are standardized and kept separate
  from the sparse TF-IDF stream, they give a small, consistent
  improvement on short/informal text and a negligible effect on
  long-form text. Naively concatenating them into the TF-IDF stream
  (with a hand-tuned weight) instead *harms* margin-based classifiers —
  an evaluation artifact, not a property of the method.

- **Semantic padding shows no genuine gain under fair evaluation.** An
  apparent large LSTM improvement is attributable to a last-timestep
  readout that prevents the zero-padding baseline from training. Under
  a fair readout where the baseline trains, the effect is null (robust
  across two readout families). The provided harness is built to make
  this kind of distinction visible.

Full per-fold numbers, significance tests, and the locked result tables
are under `results_stats/`, `results_stats_gold/` and
`results_stats_lstm/` (the Code Ocean capsule regenerates them into
`results/`): see the locked
[classical](results_stats/LOCKED_classical_results.md) and
[LSTM](results_stats_lstm/LOCKED_lstm_results.md) tables.

**Implementation notes.**

- The default base lexicon is Bing Liu's opinion lexicon (loads at 6,789
  entries, matching the paper). AFINN is also available via `load_afinn()`.
- Swap in any gensim `KeyedVectors`-compatible embedding without code changes.

## 5. License

MIT (see `LICENSE`).
