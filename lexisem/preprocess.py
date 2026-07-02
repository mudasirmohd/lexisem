"""Preprocessing: tokenisation, stopword removal, lemmatisation, POS tagging.

Mirrors the preprocessing described in Mohd et al. (2022), "Sentiment analysis
using lexico-semantic features", Journal of Information Science.
"""
from __future__ import annotations
import re
from functools import lru_cache
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s']")

# Negation cues used to build negated bigrams (from the paper).
NEGATIONS = {
    "no", "not", "rather", "won't", "wont", "never", "none", "nobody",
    "nothing", "neither", "nor", "nowhere", "cannot", "without", "n't",
}

_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stop() -> frozenset:
    return frozenset(stopwords.words("english"))


def _penn_to_wordnet(tag: str) -> str:
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("R"):
        return "r"
    return "n"


def clean_text(text: str) -> str:
    """Lowercase, strip URLs, keep alphabetic tokens and apostrophes."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str, remove_stopwords: bool = True) -> Tuple[List[str], List[str]]:
    """Return (lemmas, pos_tags) after the paper's preprocessing pipeline.

    Steps: clean -> tokenise -> POS tag -> lemmatise -> (optional) stopword removal.
    POS tags are returned before stopword removal so callers can align them if
    needed; here we filter both in lockstep.
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    if not tokens:
        return [], []
    tagged = nltk.pos_tag(tokens)
    lemmas, tags = [], []
    stop = _stop() if remove_stopwords else frozenset()
    for tok, tag in tagged:
        if tok in stop or len(tok) < 2:
            continue
        lemma = _lemmatizer.lemmatize(tok, _penn_to_wordnet(tag))
        lemmas.append(lemma)
        tags.append(tag)
    return lemmas, tags
