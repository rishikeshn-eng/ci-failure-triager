"""Learned classifier over the normalised 'error window' of a log (not the whole log: 99% of it is noise)."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .rules import ERRORISH, BENIGN, RULES, clean

_NUM = re.compile(r"0x[0-9a-f]+|\d+")


def error_window(lines: list[str], radius: int = 1, cap: int = 60) -> str:
    """Lines that look like failures (plus neighbours), numbers masked, benign lines dropped."""
    keep = set()
    for i, raw in enumerate(lines):
        ln = clean(raw)
        if (ERRORISH.search(ln) or any(r["rx"].search(ln) for r in RULES)) and not BENIGN.search(ln):
            keep.update(range(max(0, i - radius), min(len(lines), i + radius + 1)))
    sel = [clean(lines[i]) for i in sorted(keep)][-cap:]
    return "\n".join(_NUM.sub("N", l) for l in sel)


def train(runs):
    X = [error_window(r["lines"]) for r in runs]
    y = [r["label"] for r in runs]
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=500, C=4.0)).fit(X, y)


def predict(model, run) -> tuple[str, float]:
    p = model.predict_proba([error_window(run["lines"])])[0]
    i = p.argmax()
    return model.classes_[i], float(p[i])
