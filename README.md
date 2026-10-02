# ci-failure-triager

Classify a failed CI run as **flaky test**, **dependency**, **infrastructure** or **real bug**, and return the log lines that prove it.

**Live demo:** https://rishikeshn-eng.github.io/ci-failure-triager/ (paste a failed log and get a verdict in your browser; same `rules.json` as the Python pipeline)

> **Status: evaluated on synthetic logs, not on GHALogs.**
> GHALogs (513k runs, 142 GB) was not downloaded. `citriage/synth.py` generates GitHub-Actions-style logs (timestamp prefixes, step groups, 60-320 lines of build noise, decoys such as `npm WARN ... 404` and `test_timeout_handler PASSED`) with a known label and a planted proof line. The numbers below measure the pipeline against that generator. They are not an estimate of accuracy on real CI logs, and the rules and the generator were written by the same person, which flatters the rules.

## Pipeline (`citriage/triage.py`)

1. **Rules** (`rules.json`, 18 signatures): regex families per class, ignoring benign lines (warnings, passing tests). A class's score is its best rule weight plus a small bonus for corroborating rules. If one class owns at least 80% of the score, the rules decide, and the matched lines are the evidence.
2. **ML** (`ml.py`): TF-IDF + logistic regression on the *error window* only (failure-looking lines plus neighbours, numbers masked), because ~99% of a CI log is noise. Used when rules are ambiguous and the model is at least 60% sure.
3. **Gemini** (`llm.py`): for runs neither stage can decide. Sends only the extracted window, never the whole log, and rejects outputs with an invalid label. Implemented and unit-tested against a fake transport; **not run live** (no API key at build time).

## Results (1,200 held-out synthetic runs per row)

| test set | rules only | ML only | rules > ML > unsure | runs left "unsure" (would go to Gemini) |
|---|---|---|---|---|
| seen families | 88.5% | 86.7% | 86.8% | 17.6% |
| **novel families** (signatures the ML never saw) | 69.3% | **31.2%** | 71.6% | 33.0% |
| all families | 85.1% | 78.7% | 84.2% | 19.1% |

- About 18% of synthetic runs contain no signature at all (only `exit code 1`), so accuracy tops out near 88%. Rules at 88.5% on seen families are at that ceiling, so that row says little.
- The informative row is **novel families**: the learned model collapses to 31% when a failure looks different from its training data, while broad hand-written signatures degrade gracefully. That is the argument for a small, validated LLM fallback on exactly those runs. Whether Gemini actually closes the gap is untested here.
- Evidence hit (a returned line is the planted proof line): 100% on seen families, 75% on novel ones (rules miss the signature entirely).
- Flaky vs real bug is the hard distinction in practice: flakiness is defined by whether a rerun passes, which a single log cannot show. The synthetic data plants retry/timeout/port signatures; real flaky tests often look exactly like real failures. Expect a much lower number on real data.

## Run

```bash
pip install scikit-learn pytest
python -m pytest -q
python -m citriage.evaluate        # writes docs/results.json
python scripts/build_site.py       # docs/index.html
```

### Next step: real labels

To evaluate on real logs, sample GHALogs (or pull failed runs via `gh api repos/{owner}/{repo}/actions/runs/{id}/logs`), label a few hundred by hand, and swap them in for `synth.generate`. The classifier takes `{"lines": [...], "label": ...}` dicts.
