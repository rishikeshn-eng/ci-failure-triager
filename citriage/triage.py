"""Three-stage triage: rules -> ML -> (optional) Gemini. Each stage only runs when the previous is unsure."""
from __future__ import annotations

from . import ml, rules

RULE_CONF = 0.8    # rules decide alone when one class owns >= 80% of the matched score
ML_CONF = 0.6      # below this the run is escalated


def triage(run, model, llm=None):
    r = rules.classify(run["lines"])
    if r["matched"] and r["confidence"] >= RULE_CONF:
        return {"label": r["label"], "stage": "rules", "confidence": r["confidence"], "evidence": r["evidence"]}
    label, p = ml.predict(model, run) if model is not None else (r["label"], 0.0)
    if model is not None and p >= ML_CONF:
        ev = r["evidence"]
        return {"label": label, "stage": "ml", "confidence": p, "evidence": ev}
    if llm is not None:
        window = ml.error_window(run["lines"]).split("\n")
        g = llm.classify(window)
        if g:
            return {"label": g["label"], "stage": "llm", "confidence": 1.0,
                    "evidence": [{"line_no": None, "text": g["evidence"], "rule": "llm", "label": g["label"]}]}
    return {"label": label, "stage": "unsure", "confidence": p, "evidence": r["evidence"]}
