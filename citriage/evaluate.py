from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score

from . import ml, rules, synth
from .triage import triage

LABELS = rules.LABELS


def score(runs, preds):
    y = [r["label"] for r in runs]
    p = [x["label"] for x in preds]
    acc = sum(a == b for a, b in zip(y, p)) / len(y)
    withp = [(r, x) for r, x in zip(runs, preds) if r["proof"]]
    hit = sum(any(e["line_no"] in r["proof"] for e in x["evidence"] if e["line_no"] is not None) for r, x in withp) / max(1, len(withp))
    return {"accuracy": acc, "macro_f1": float(f1_score(y, p, average="macro", labels=LABELS)),
            "evidence_hit": hit, "stages": dict(Counter(x["stage"] for x in preds)),
            "confusion": confusion_matrix(y, p, labels=LABELS).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--llm", choices=["none", "gemini"], default="none", help="gemini: escalate 'unsure' runs (needs GEMINI_API_KEY)")
    ap.add_argument("--out", default="docs/results.json")
    a = ap.parse_args()
    train = synth.generate(a.n, seed=1, families="seen")
    model = ml.train(train)
    suites = {"seen families": synth.generate(1200, seed=2, families="seen"),
              "novel families": synth.generate(1200, seed=3, families="novel"),
              "all families": synth.generate(1200, seed=4, families="all")}
    llm = None
    if a.llm == "gemini":
        from .llm import GeminiTriage
        llm = GeminiTriage()
    res = {}
    for name, runs in suites.items():
        row = {}
        row["rules only"] = score(runs, [dict(label=(c := rules.classify(r["lines"]))["label"], stage="rules",
                                              evidence=c["evidence"]) for r in runs])
        row["ml only"] = score(runs, [dict(label=ml.predict(model, r)[0], stage="ml", evidence=rules.classify(r["lines"])["evidence"]) for r in runs])
        row["rules > ml > unsure"] = score(runs, [triage(r, model) for r in runs])
        if llm is not None:
            c0 = llm.calls
            row["rules > ml > gemini"] = score(runs, [triage(r, model, llm) for r in runs])
            row["rules > ml > gemini"]["llm"] = {"calls": llm.calls - c0, "rejected": llm.rejected, "tokens_in": llm.tokens_in, "tokens_out": llm.tokens_out}
        res[name] = row
        print(name, {k: (round(v["accuracy"], 3), round(v["evidence_hit"], 3)) for k, v in row.items()})
    # ceiling: runs without any planted signature are undecidable
    ceiling = {n: 1 - sum(1 for r in runs if not r["proof"]) / len(runs) * 0.65 for n, runs in suites.items()}
    # sample runs for the UI (all families) with the full pipeline's verdict
    demo = []
    for r in suites["all families"][:60]:
        t = triage(r, model)
        err = [i for i, l in enumerate(r["lines"]) if i in {e["line_no"] for e in t["evidence"] if e["line_no"] is not None}]
        lo = max(0, (min(err) if err else len(r["lines"]) - 6) - 3)
        hi = min(len(r["lines"]), (max(err) if err else len(r["lines"])) + 4)
        demo.append({"run_id": r["run_id"], "truth": r["label"], "pred": t["label"], "stage": t["stage"],
                     "conf": round(t["confidence"], 2), "n_lines": len(r["lines"]),
                     "evidence": [e["line_no"] for e in t["evidence"] if e["line_no"] is not None],
                     "excerpt": [{"n": i, "t": r["lines"][i][:200]} for i in range(lo, hi)]})
    Path(a.out).write_text(json.dumps({"suites": res, "ceiling": ceiling, "demo": demo,
                                       "rules": json.loads((Path(__file__).parent / "rules.json").read_text()),
                                       "train_runs": len(train)}))


if __name__ == "__main__":
    main()
