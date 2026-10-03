"""Evaluate the triager on real failed-job logs labelled in data/real_labels.csv.

    python scripts/collect_real_logs.py --out data/real_logs        # re-fetch (GitHub keeps logs ~90 days)
    python scripts/eval_real.py data/real_logs [--old-rules rules_v1.json]

Labels: flaky_test | dependency | infrastructure | real_bug | other ("other" = policy checks, bot/process failures, or
too ambiguous to call; a good triager should abstain on these). One annotator (Claude), read from log excerpts.
"""
import csv, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from citriage import ml, rules, synth
from citriage.triage import triage

logs = sys.argv[1]
old = sys.argv[sys.argv.index("--old-rules") + 1] if "--old-rules" in sys.argv else None
lab = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), "..", "data", "real_labels.csv"))))
model = ml.train(synth.generate(3000, seed=1, families="seen"))   # trained on synthetic logs only
four = ["flaky_test", "dependency", "infrastructure", "real_bug"]


def load(r):
    fn = f"{r['repo'].replace('/', '__')}__{r['job_id']}.log"
    return open(os.path.join(logs, fn), errors="replace").read().splitlines()


def run(name, rl):
    rows = []
    for r in lab:
        lines = load(r)
        c = rules.classify(lines, rules=rl)
        t = triage({"lines": lines}, model) if rl is None else None
        rows.append((r, c))
    return rows


def report(name, rows, subset):
    rs = [(r, c) for r, c in rows if subset(r)]
    l4 = [(r, c) for r, c in rs if r["label"] in four]
    oth = [(r, c) for r, c in rs if r["label"] == "other"]
    forced = sum(c["label"] == r["label"] for r, c in l4) / len(l4)                      # unmatched -> real_bug (old behaviour)
    abst = sum((c["label"] if c["matched"] else "unknown") == r["label"] for r, c in l4) / len(l4)
    cov = sum(c["matched"] for r, c in l4) / len(l4)
    prec = sum(c["label"] == r["label"] for r, c in l4 if c["matched"]) / max(1, sum(c["matched"] for r, c in l4))
    ok_other = sum(not c["matched"] for r, c in oth) / max(1, len(oth))
    allacc = (sum((c["label"] if c["matched"] else "unknown") == r["label"] for r, c in l4) + sum(not c["matched"] for r, c in oth)) / len(rs)
    maj = Counter(r["label"] for r, c in l4).most_common(1)[0]
    print(f"{name:28s} n={len(rs)} (4-class {len(l4)}, other {len(oth)}) | forced-label acc {forced:.3f} | answer-or-abstain acc {abst:.3f} | "
          f"coverage {cov:.2f} precision-when-answering {prec:.3f} | abstains on other {ok_other:.2f} | all-81 acc {allacc:.3f} | majority '{maj[0]}' {maj[1]/len(l4):.3f}")
    return dict(n=len(rs), n4=len(l4), n_other=len(oth), forced_acc=forced, answer_or_abstain_acc=abst, coverage=cov,
                precision_when_answering=prec, abstains_on_other=ok_other, overall_acc=allacc, majority=maj[1] / len(l4))


out = {}
variants = [("extended rules", None)] + ([("original rules", rules.load_rules(old))] if old else [])
for name, rl in variants:
    rows = run(name, rl)
    out[name] = {s: report(f"{name} [{s}]", rows, f) for s, f in (("all", lambda r: True), ("test half", lambda r: r["split"] == "test"), ("dev half", lambda r: r["split"] == "dev"))}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "docs", "real_results.json"), "w"), indent=1)
