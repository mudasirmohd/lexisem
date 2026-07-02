"""Sentiment lexicon loading and semantic expansion.

Implements Algorithm 1 from Mohd et al. (2022):
  1. Lemmatise lexicon entries.
  2. For each entry, retrieve semantically similar words from a distributional
     model; keep those with similarity > threshold (paper uses 0.75).
  3. Add WordNet antonyms (polarity-flipped) and hyponyms.

Two base lexicons are supported: Bing Liu's opinion lexicon (via NLTK) and
AFINN. Both are freely available and OSI/CC compatible for research use.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

_lemma = WordNetLemmatizer()


@dataclass
class Lexicon:
    positive: Set[str] = field(default_factory=set)
    negative: Set[str] = field(default_factory=set)

    def polarity(self, word: str) -> int:
        if word in self.positive:
            return 1
        if word in self.negative:
            return -1
        return 0

    def __len__(self) -> int:
        return len(self.positive) + len(self.negative)


def load_bing_liu() -> Lexicon:
    nltk.download("opinion_lexicon", quiet=True)
    from nltk.corpus import opinion_lexicon
    return Lexicon(
        positive={w.lower() for w in opinion_lexicon.positive()},
        negative={w.lower() for w in opinion_lexicon.negative()},
    )


def load_afinn() -> Lexicon:
    from afinn import Afinn  # afinn ships its own word list
    af = Afinn()
    pos, neg = set(), set()
    for word, score in af._dict.items():
        if score > 0:
            pos.add(word.lower())
        elif score < 0:
            neg.add(word.lower())
    return Lexicon(positive=pos, negative=neg)


def _antonyms(word: str) -> Set[str]:
    out = set()
    for syn in wn.synsets(word):
        for lem in syn.lemmas():
            for ant in lem.antonyms():
                out.add(ant.name().replace("_", " ").lower())
    return out


def _hyponyms(word: str) -> Set[str]:
    out = set()
    for syn in wn.synsets(word, pos=wn.NOUN):
        for hyp in syn.hyponyms():
            for lem in hyp.lemmas():
                out.add(lem.name().replace("_", " ").lower())
    return out


def expand_lexicon(
    base: Lexicon,
    kv,
    sim_threshold: float = 0.75,
    topn: int = 10,
    use_wordnet: bool = True,
) -> Lexicon:
    """Expand a base lexicon using distributional model `kv` + WordNet.

    `kv` is a gensim-style vectors object (e.g. FastText.wv) exposing
    `.most_similar` and `.key_to_index`.
    """
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    pos = {_lemma.lemmatize(w) for w in base.positive}
    neg = {_lemma.lemmatize(w) for w in base.negative}

    def _semantic(seed_words: Set[str]) -> Set[str]:
        added = set()
        for w in seed_words:
            try:
                for cand, score in kv.most_similar(w, topn=topn):
                    if score > sim_threshold and cand.isalpha():
                        added.add(cand.lower())
            except KeyError:
                continue  # OOV even with subwords (rare for FastText)
        return added

    exp_pos = set(pos) | _semantic(pos)
    exp_neg = set(neg) | _semantic(neg)

    if use_wordnet:
        # Antonyms flip polarity: antonym of a positive word -> negative, etc.
        for w in list(pos):
            exp_neg |= _antonyms(w)
        for w in list(neg):
            exp_pos |= _antonyms(w)
        # Hyponyms preserve polarity.
        for w in list(pos):
            exp_pos |= _hyponyms(w)
        for w in list(neg):
            exp_neg |= _hyponyms(w)

    # Resolve conflicts: a word claimed by both sides is dropped (ambiguous).
    overlap = exp_pos & exp_neg
    exp_pos -= overlap
    exp_neg -= overlap
    return Lexicon(positive=exp_pos, negative=exp_neg)


def expansion_stats(base: Lexicon, expanded: Lexicon) -> Dict[str, int]:
    return {
        "base_positive": len(base.positive),
        "base_negative": len(base.negative),
        "base_total": len(base),
        "expanded_positive": len(expanded.positive),
        "expanded_negative": len(expanded.negative),
        "expanded_total": len(expanded),
    }
