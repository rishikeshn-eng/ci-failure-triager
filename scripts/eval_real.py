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
    """Read the cached log, or fetch it by job id (GitHub keeps logs ~90 days)."""
    import re, subprocess
    fn = os.path.join(logs, f"{r['repo'].replace('/', '__')}__{r['job_id']}.log")
    if not os.path.exists(fn):
        os.makedirs(logs, exist_ok=True)
        p = subprocess.run(["gh", "api", "--allow-escape-sequences", f"repos/{r['repo']}/actions/jobs/{r['job_id']}/logs"],
                           capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise SystemExit(f"cannot fetch {r['repo']} job {r['job_id']}: logs expired? {p.stderr[:100]}")
        open(fn, "w").write(re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", p.stdout))
    return open(fn, errors="replace").read().splitlines()


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


if "--gemini" in sys.argv:
    from citriage.llm import GeminiTriage
    llm = GeminiTriage()
    g = {"gemini_only": [], "pipeline": []}
    for r in lab:
        lines = load(r)
        w = ml.error_window(lines).split("\n")
        v = llm.classify(w)
        g["gemini_only"].append((r["label"], v["label"] if v else "unknown"))
        t = triage({"lines": lines}, model, llm)
        g["pipeline"].append((r["label"], t["label"] if t["stage"] != "unsure" else "unknown", t["stage"]))
    def summ(pairs):
        four_ = [(a, b) for a, b, *_ in pairs if a in four]
        oth = [(a, b) for a, b, *_ in pairs if a == "other"]
        return {"n4": len(four_), "acc4": sum(a == b for a, b in four_) / len(four_),
                "answered4": sum(b not in ("unknown", "other") for a, b in four_) / len(four_),
                "precision_when_answering": sum(a == b for a, b in four_ if b not in ("unknown", "other")) / max(1, sum(b not in ("unknown", "other") for a, b in four_)),
                "abstains_on_other": sum(b in ("unknown", "other") for a, b in oth) / max(1, len(oth)),
                "overall_acc": (sum(a == b for a, b in four_) + sum(b in ("unknown", "other") for a, b in oth)) / len(pairs)}
    res = {k: summ(v) for k, v in g.items()}
    res["llm"] = {"calls": llm.calls, "rejected": llm.rejected, "tokens_in": llm.tokens_in, "tokens_out": llm.tokens_out, "model": llm.model}
    res["confusion_gemini_only"] = {f"{a}>{b}": n for (a, b), n in Counter(g["gemini_only"]).items()}
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "docs", "real_gemini_results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0)

out = {}
variants = [("extended rules", None)] + ([("original rules", rules.load_rules(old))] if old else [])
for name, rl in variants:
    rows = run(name, rl)
    out[name] = {s: report(f"{name} [{s}]", rows, f) for s, f in (("all", lambda r: True), ("test half", lambda r: r["split"] == "test"), ("dev half", lambda r: r["split"] == "dev"))}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "docs", "real_results.json"), "w"), indent=1)
