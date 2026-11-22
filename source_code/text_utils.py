"""Text cleaning utilities shared by keyword extraction and embeddings."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_URL_RE = re.compile(r"http\S+|www\.\S+")
_NUM_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^a-z\s]")
_WS_RE = re.compile(r"\s+")

_STOPWORDS = None
_LEMMATIZER = None


def _ensure_nltk():
    global _STOPWORDS, _LEMMATIZER
    if _STOPWORDS is not None:
        return
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip urls/numbers/punctuation, remove stopwords, lemmatize."""
    _ensure_nltk()
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = text.replace("|", " ")
    text = _NUM_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    tokens = [t for t in text.split(" ") if t and t not in _STOPWORDS and len(t) > 2]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)
