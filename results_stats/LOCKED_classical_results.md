# LOCKED — Classical Results (lexico-semantic vs baseline)

**Protocol:** 5-fold `StratifiedKFold(shuffle=True, random_state=42)`; macro-F1 = mean(F1_pos, F1_neg); unchanged feature/model code; dense lexico-semantic features standardized and kept out of the TF-IDF branch; no per-run tuning.

| Regime | Embedding | Labels | Model | Baseline | +LexSem | Δ | t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| long-form | GloVe-100 | n/a | SVM | 0.8112 | 0.8127 | +0.0015 | 0.929 | 1.0 |
| long-form | GloVe-100 | n/a | RF | 0.7973 | 0.8005 | +0.0032 | 0.586 | 0.8125 |
| long-form | word2vec-300 | n/a | SVM | 0.8112 | 0.8193 | +0.0081 | 0.414 | 0.4375 |
| long-form | word2vec-300 | n/a | RF | 0.7973 | 0.8073 | +0.0099 | 0.287 | 0.3125 |
| short/informal, heuristic labels | GloVe-100 | emoticon (distant) | SVM | 0.7558 | 0.7623 | +0.0065 | 0.019 | 0.0625 |
| short/informal, heuristic labels | GloVe-100 | emoticon (distant) | RF | 0.7287 | 0.7339 | +0.0052 | 0.153 | 0.1875 |
| short/informal, GOLD labels | GloVe-100 | human | SVM | 0.8130 | 0.8407 | +0.0278 | 0.116 | 0.1875 |
| short/informal, GOLD labels | GloVe-100 | human | RF | 0.7320 | 0.7666 | +0.0346 | 0.113 | 0.1875 |
| short/informal, GOLD labels | word2vec-300 | human | SVM | 0.8130 | 0.8151 | +0.0022 | 0.928 | 1.0 |
| short/informal, GOLD labels | word2vec-300 | human | RF | 0.7320 | 0.7924 | +0.0603 | 0.136 | 0.3125 |

## Pooled (per config, both models, 10 fold-deltas)

| Config | mean Δ | positive | one-sample t p | Wilcoxon p |
|---|---|---|---|---|
| glove_movie | +0.0024 | 5/10 | 0.7764 | 0.8457 |
| w2v_movie | +0.0090 | 7/10 | 0.1471 | 0.1934 |
| glove_twitter | +0.0059 | 9/10 | 0.0057 | 0.0195 |
| sent140_glove | +0.0312 | 8/10 | 0.0153 | 0.0273 |
| sent140_w2v | +0.0312 | 5/10 | 0.1703 | 0.3750 |

**Gold overall (20 deltas, 4/4 cells positive):** mean Δ=+0.0312, t_p=0.01304, Wilcoxon p=0.02395, sign-test p=0.2632 (note: sign-test n.s. — pooled parametric/rank significance not robust to non-independence).

## Interpretation (corrected)

- **Regime story.** Effect is **negligible on long-form** (movie_reviews, Δ≈0.001–0.010, n.s.), **small-but-present on short/informal heuristic-labeled** text (twitter_samples, Δ≈+0.006, SVM cell t_p=0.019), and **largest on short/informal GOLD** text (Sentiment140, Δ≈+0.03–0.06).
- **Gold short-text.** Large effect sizes (+0.03–0.06), **significant when pooled** (GloVe-gold config t_p=0.015; gold overall t_p=0.013/W_p=0.024), but **individually per-cell underpowered** (t_p 0.11–0.93) due to small n=359 and high fold variance.
- **RETIRED claim.** 'Effect scales with embedding quality' is **NOT supported**: it held on movie/heuristic-twitter but reversed on gold (GloVe significant, word2vec noisy/underpowered).
- **Methodological headline.** Naïve concatenation (dense features through TF-IDF + hard 3× weight) **harmed the SVM (~−0.05)**; the principled fix removes the harm everywhere and yields the short-text-concentrated positive effect above.
