"""Text preprocessing for the spam classifier.

This is the same pipeline the notebook builds cell by cell, extracted into an
importable module so the Streamlit app and the test suite share one
implementation instead of drifting apart.

Pipeline: lowercase -> tokenise -> drop non-alphanumeric -> drop stopwords ->
Porter stem -> join.
"""

import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

_NLTK_DATA = (("corpora/stopwords", "stopwords"), ("tokenizers/punkt_tab", "punkt_tab"))


def ensure_nltk_data() -> None:
    """Download the NLTK corpora this module needs, if they are missing."""
    for path, package in _NLTK_DATA:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def _load_stopwords() -> frozenset:
    """Load the English stopword list, downloading it if it is absent.

    This runs at import time, which means a fresh container or CI runner that
    has never used NLTK before would crash on ``import preprocessing`` with a
    LookupError. Downloading on demand makes the module importable anywhere
    without every caller having to remember to prime NLTK first.
    """
    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("english"))


# Built once at import time. The original code called stopwords.words('english')
# inside the per-token loop, which rebuilt the whole stopword list for every
# single word of every message.
STOPWORDS = _load_stopwords()
PUNCTUATION = frozenset(string.punctuation)
STEMMER = PorterStemmer()


def transform_text(text: str) -> str:
    """Normalise a raw message into the string the vectoriser expects.

    Args:
        text: The raw message, as typed by the user.

    Returns:
        Lower-cased, stopword-free, stemmed words joined by single spaces.
    """
    tokens = nltk.word_tokenize(text.lower())

    kept: List[str] = [
        token
        for token in tokens
        if token.isalnum() and token not in STOPWORDS and token not in PUNCTUATION
    ]

    return " ".join(STEMMER.stem(token) for token in kept)
