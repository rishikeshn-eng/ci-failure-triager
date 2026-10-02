import json

from citriage import ml, rules, synth
from citriage.llm import GeminiTriage
from citriage.triage import triage


def L(*lines):
    return [f"2025-03-01T12:00:{i:02d}.0000000Z {l}" for i, l in enumerate(lines)]


def test_each_class_signature_is_found_with_proof_line():
    cases = {
        "dependency": "npm ERR! code E404",
        "infrastructure": "##[error]The runner has received a shutdown signal.",
        "flaky_test": "Error: listen EADDRINUSE: address already in use :::3000",
        "real_bug": "src/a.ts(3,1): error TS2322: Type 'string' is not assignable to type 'number'.",
    }
    for label, line in cases.items():
        r = rules.classify(L("step", "noise", line, "##[error]Process completed with exit code 1."))
        assert r["label"] == label and r["evidence"][0]["line_no"] == 2, (label, r)


def test_benign_lines_with_failure_words_do_not_trigger():
    r = rules.classify(L("npm WARN deprecated foo: 404 Not Found - no longer supported",
                         "test_timeout_handler PASSED", "retrying download (1/3)", "0 failed, 12 passed"))
    assert not r["matched"]


def test_generic_failure_falls_back_with_zero_confidence():
    r = rules.classify(L("build", "##[error]Process completed with exit code 1."))
    assert not r["matched"] and r["confidence"] == 0 and r["evidence"]


def test_rules_json_is_valid_in_python_and_js_syntax_subset():
    for r in json.loads(open(rules.__file__.replace("rules.py", "rules.json")).read()):
        assert {"label", "name", "w", "pattern"} <= set(r) and r["label"] in rules.LABELS
        assert "(?<" not in r["pattern"] and "(?P" not in r["pattern"]  # keep JS-compatible


def test_pipeline_beats_chance_and_escalates_only_when_unsure():
    model = ml.train(synth.generate(600, seed=1, families="seen"))
    runs = synth.generate(300, seed=2)
    preds = [triage(r, model) for r in runs]
    acc = sum(p["label"] == r["label"] for p, r in zip(preds, runs)) / len(runs)
    assert acc > 0.75
    assert {p["stage"] for p in preds} <= {"rules", "ml", "unsure"}  # no LLM configured


def test_gemini_escalation_used_and_bad_output_rejected():
    def good(body):
        return {"candidates": [{"content": {"parts": [{"text": json.dumps({"label": "dependency", "evidence_line": 1})}]}}]}

    def bad(body):
        return {"candidates": [{"content": {"parts": [{"text": json.dumps({"label": "gremlins"})}]}}]}
    g = GeminiTriage(transport=good)
    assert g.classify(["a", "npm ERR! 404"]) == {"label": "dependency", "evidence": "npm ERR! 404"}
    b = GeminiTriage(transport=bad)
    assert b.classify(["a"]) is None and b.rejected == 1
    run = {"lines": L("x", "weird failure", "##[error]Process completed with exit code 1."), "label": "dependency"}
    out = triage(run, None, llm=g)
    assert out["stage"] == "llm" and out["label"] == "dependency"
