# LOCKED — LSTM Results (zero- vs semantic-padding)

**Claim under test (paper §3.4.2):** replacing zero padding with "semantic
padding" (sentiment-bearing word vectors in the pad slots) improves an LSTM
sentiment classifier.

**Protocol:** 5-fold `StratifiedKFold(shuffle=True, random_state=42)`;
macro-F1 = mean(F1_pos, F1_neg) per fold; paired t-test + Wilcoxon on the 5
per-fold deltas (semantic − zero). Within each fold both conditions share the
same fold split, weight init, minibatch order and dropout sequence (one
per-fold seed), so the **only** difference is what fills the pad slots. 4-layer
LSTM, hidden=128, dropout=0.2, Adam lr=5e-4, gradient clip (max-norm 5.0),
epochs=10, max_len=300, GloVe-100. A fold whose eval predictions are all one
class is a "collapse"; it is retried once with a perturbed seed and, if it
still collapses, **flagged** (never silently averaged). Labels reconstructed
deterministically (source CSVs cached only as tokens): movie = NLTK
movie_reviews pos(1000)+neg(1000); twitter = NLTK twitter_samples
pos(5000)+neg(5000), emoticon distant labels.

The comparison is run under **two readout families**, because the readout is
not neutral to this method — it interacts with *where* the padding sits:

- **meanpool** — post-pad (real tokens first); masked mean-pool over LSTM
  outputs: zero-pad pools over real-token positions only, semantic pools over
  all slots (its pads are content by design). Lets the zero baseline train and
  measures whether the sentiment-filled pad region adds signal.
- **prelast** — pre-pad (real tokens **last**); last-time-step readout. This is
  semantic padding's **most favorable legitimate** readout: the sentiment pads
  are consumed sequentially *before* the real tokens (priming the LSTM state),
  and the last time-step still sees a real token so the zero baseline trains.

## Results

| Readout | Dataset | zero macro-F1 | semantic macro-F1 | Δ (sem−zero) | pos folds | t p | Wilcoxon p | collapses (z/s) |
|---|---|---|---|---|---|---|---|---|
| meanpool | movie (long) | 0.7633 ± 0.0407 | 0.7773 ± 0.0342 | **+0.0140** | 5/5 | 0.037 | 0.0625 | 0/0 |
| meanpool | twitter (short) | 0.7268 ± 0.0089 | 0.5011 ± 0.1413 | **−0.2256** | 0/5 | 0.026 | 0.0625 | 0/1 |
| prelast | movie (long) | 0.6405 ± 0.1420 | 0.6612 ± 0.0727 | **+0.0207** | 3/5 | 0.798 | 1.0 | 0/0 |
| prelast | twitter (short) | 0.7241 ± 0.0094 | 0.7273 ± 0.0081 | **+0.0032** | 3/5 | 0.214 | 0.3125 | 0/0 |

## Interpretation (honest, no spin)

- **The paper's core deep-learning claim does not reproduce under fair
  evaluation.** In no cell is semantic padding a *robust, significant*
  improvement. The only nominally-significant positive — meanpool/movie
  (+0.014, t_p=0.037) — is (i) trivially small, (ii) an artifact of long-form
  docs: movie reviews exceed max_len, so most docs have **no** pad slots and
  semantic ≡ zero (fold 1 Δ is exactly 0.0000); the tiny effect comes only from
  the minority of shorter docs, and (iii) not robust — it flips to a large
  *significant loss* on meanpool/twitter and vanishes on prelast/movie
  (+0.021, n.s., high variance incl. a near-dead fold).

- **Decisive clean cell: prelast / twitter.** Short-text is the regime where
  the *classical* lexico-semantic effect lived, and prelast is semantic
  padding's most favorable legitimate readout. There, both conditions train
  cleanly (~0.72–0.73, no collapse) and Δ = **+0.0032 (n.s., t_p=0.214)** —
  essentially zero. Semantic padding neither helps nor hurts once the baseline
  can actually train.

- **The apparent benefit was a readout artifact.** A naive post-pad +
  last-time-step setup gives twitter Δ ≈ **+0.20 (t_p≈0)** — but only because
  the *zero* baseline collapses to one class on every fold (0.3333 macro-F1):
  signal cannot survive ~285 trailing zero-input steps to the last-time-step
  readout. Semantic padding "wins" solely by comparison to a broken baseline.
  Give the baseline a fair readout and the gap disappears (prelast) or reverses
  (meanpool). See RETIRED claim below.

- **Why meanpool *hurts* on short text.** `semantic_padding`
  (`features.py:106-107`) fills pad slots with
  `present + sorted(lex.positive) + sorted(lex.negative)` — for a ~15-token
  tweet padded to 300 that is ~285 slots drawn from the **same
  alphabetically-sorted global lexicon list for every document**. Mean-pooling
  makes ~93% of the pooled vector a document-independent constant, drowning the
  real signal → near-collapse (the 0.33/0.42/0.46 folds). This is a property of
  the padding content, not a tuning issue.

- **Robust to readout choice.** "No genuine gain from semantic padding" holds
  under **both** readout families (meanpool: null-to-negative; prelast: null).
  The conclusion does not depend on how the sequence is read out.

### RETIRED claim
'Semantic padding improves the LSTM (Δ≈+0.20 on short text).' **NOT supported.**
That result required a last-time-step readout that collapses the zero-padding
baseline to a single class (macro-F1 0.3333 on 5/5 twitter folds). Under any
readout where the baseline trains, the effect is null (prelast, +0.003 n.s.) or
negative (meanpool, −0.226). The +0.20 is a readout artifact, not a
representational gain.

## Relationship to the classical results
This mirrors the classical correction. There, naive concatenation (dense
features through TF-IDF + a hard 3× weight) inflated an apparent effect that a
principled pipeline erased. Here, a naive last-time-step readout inflates an
apparent deep-learning effect that a fair readout erases. In both paths the
faithful, principled evaluation removes an artifact — and for the LSTM path the
honest result is a **null**: semantic padding does not improve the classifier.
