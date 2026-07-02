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

## 5. License

MIT (see `LICENSE`).
