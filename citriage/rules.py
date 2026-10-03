"""Signature rules shared with the in-browser demo (rules.json). Scores each class and extracts proof lines."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

LABELS = ["flaky_test", "dependency", "infrastructure", "real_bug"]
def load_rules(path) -> list[dict]:
    return [dict(r, rx=re.compile(r["pattern"], re.I)) for r in json.loads(Path(path).read_text())]


RULES = load_rules(Path(__file__).parent / "rules.json")
TS = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ")
# lines that are announcing a failure, used when no signature matches
ERRORISH = re.compile(r"##\[error\]|\bERROR\b|\bFAIL(ED)?\b|\bfatal\b|exception|panic|exit code [1-9]", re.I)
# informational lines that merely contain failure-looking words
# command echoes, shell script bodies and job titles show up in logs but are not outcomes
NOT_OUTCOME = re.compile(r"^\s*(echo|if|elif|else|fi|g\+\+|gcc|clang|c\+\+)\b|^\s*(curl|wget)\s+(-|http)|^\s*cc\s+-|^Complete job name|^##\[group\]|(?-i:^\s+[A-Z][A-Z0-9_]*:\s)|(?-i:^[A-Z][A-Z0-9_]*=)|^\s*-C\S|\[command\]", re.I)
BENIGN = re.compile(r"npm WARN|deprecated|PASSED|PASS |passed|\bok\b|0 failed|0 errors|skipp|retrying download \(\d/\d\)$|::debug::", re.I)


def clean(line: str) -> str:
    return TS.sub("", line).rstrip()


def classify(lines: list[str], k: int = 3, rules: list[dict] | None = None) -> dict:
    """Returns {label, confidence, scores, evidence:[{line_no, text, rule, label}]}.

    A line scores a rule only if it is not benign. A class's score is its best rule weight plus a small bonus per
    additional distinct rule (capped). Confidence = share of the top class in the total score.
    """
    hits = []  # (label, rule name, weight, idx)
    for i, raw in enumerate(lines):
        ln = clean(raw)
        if (BENIGN.search(ln) or NOT_OUTCOME.search(ln)) and not ln.startswith("##[error]"):
            continue
        for r in (rules or RULES):
            if r["rx"].search(ln):
                hits.append((r["label"], r["name"], r["w"], i))
    by = defaultdict(dict)
    for lab, name, w, i in hits:
        by[lab].setdefault(name, (w, i))
    scores = {lab: 0.0 for lab in LABELS}
    for lab, d in by.items():
        ws = sorted((w for w, _ in d.values()), reverse=True)
        scores[lab] = ws[0] + 0.15 * min(2, len(ws) - 1)
    total = sum(scores.values())
    if total == 0:
        err = [i for i, l in enumerate(lines) if ERRORISH.search(clean(l))]
        ev = [{"line_no": i, "text": clean(lines[i])[:240], "rule": "generic error", "label": None} for i in err[-k:]]
        return {"label": "real_bug", "confidence": 0.0, "scores": scores, "evidence": ev, "matched": False}
    label = max(scores, key=scores.get)
    conf = scores[label] / total
    ev = sorted(((w, i, name) for name, (w, i) in by[label].items()), reverse=True)[:k]
    evidence = [{"line_no": i, "text": clean(lines[i])[:240], "rule": name, "label": label} for w, i, name in sorted(ev, key=lambda t: t[1])]
    return {"label": label, "confidence": conf, "scores": scores, "evidence": evidence, "matched": True}
