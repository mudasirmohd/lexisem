"""lexisem: reference implementation of lexico-semantic features for sentiment analysis.

Reference: Mohd, Javeed, Nowsheena, Wani, Khanday (2022), "Sentiment analysis
using lexico-semantic features", Journal of Information Science,
DOI:10.1177/01655515221124016.
"""
from .lexicon import (
    Lexicon, load_bing_liu, load_afinn, expand_lexicon, expansion_stats,
)
from .embeddings import load_vectors
from .features import (
    extract, semantic_padding, surface_features, lexico_semantic_features,
)
from .preprocess import preprocess

__version__ = "1.0.0"
__all__ = [
    "Lexicon", "load_bing_liu", "load_afinn", "expand_lexicon", "expansion_stats",
    "load_vectors", "extract", "semantic_padding",
    "surface_features", "lexico_semantic_features", "preprocess",
]
