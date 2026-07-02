"""Offline-safe NLTK resource loading.

`nltk.download(name)` always tries the network first, which spams
"Connection refused" errors in an offline capsule even when the resource is
already present (e.g. pre-staged under NLTK_DATA). `ensure_nltk` looks the
resource up first via `nltk.data.find` and only downloads if it is genuinely
missing.
"""
from __future__ import annotations

import nltk

# Standard NLTK sub-directories a resource may live under.
_KINDS = ("corpora", "tokenizers", "taggers", "sentiment", "chunkers")


def ensure_nltk(*names: str) -> None:
    for name in names:
        # Some resources stay zipped after download (e.g. wordnet.zip); NLTK's
        # corpus loader reads them fine, so probe both the bare and .zip forms.
        if any(_present(f"{kind}/{name}") or _present(f"{kind}/{name}.zip")
               for kind in _KINDS):
            continue
        nltk.download(name, quiet=True)


def _present(path: str) -> bool:
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False
