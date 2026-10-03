# ci-failure-triager

Classify a failed CI run as **flaky test**, **dependency**, **infrastructure** or **real bug**, and return the log lines that prove it.

**Live demo:** https://rishikeshn-eng.github.io/ci-failure-triager/ (paste a failed log and get a verdict in your browser; same `rules.json` as the Python pipeline)

> **Status: two evaluations, and the real one is much harder.** (1) Synthetic GitHub-Actions-style logs with planted labels (`citriage/synth.py`): flattering, mostly a pipeline test. (2) **81 real failed-job logs from 27 public repos** (home-assistant, vscode, react, rust, node, numpy, pandas, airflow, django, angular, next.js, deno, grafana, go, pytorch, transformers, scikit-learn, tokio, prometheus, terraform, electron, godot, flask, requests, TypeScript, vue), collected through the GitHub API (`scripts/collect_real_logs.py`) and labelled by **one annotator (me, Claude)** from log excerpts: `data/real_labels.csv`. GHALogs itself (513k runs, 142 GB) was not used. The logs are not committed (third-party content; GitHub keeps them ~90 days), only labels and job ids.

## Real-log results

Of the 81 logs, 53 fit the four classes (26 real bug, 13 infrastructure, 11 dependency, 3 flaky) and **28 are `other`**: commit-message / changelog / PR-title policy checks, bot and process failures, or too ambiguous to call. Real CI failures turned out to be mostly lint, formatting, type-check, Dependabot and bot jobs, which the synthetic generator barely had.

| rules | accuracy, forced label | coverage | precision when it answers | abstains on the 28 `other` |
|---|---|---|---|---|
| Original rules (written before seeing real logs) | **56.6%** | 43% | 65% | 61% |
| Extended rules | **86.8%** | 81% | 88% | 54% |
| Always guess `real_bug` | 49.1% | 100% | 49% | n/a |

- "Forced label" counts a no-match as `real_bug`, as the original pipeline did. The original rules barely beat guessing `real_bug` on real logs (56.6% against 49.1%), against 85-88% on synthetic ones.
- **The extended rules are not a clean result.** I wrote the 13 added patterns (Dependabot errors, artifact-not-found, auth/secrets failures, lint-tool failures, process-pool crashes...) *after reading all 81 logs*. A dev/test split by index gives 85.7% / 88.0%, but both halves come from the same repos, so it is not independent. Treat 56.6% as the untouched number and 86.8% as an upper bound on what the rules would do on new repos.
- **Evidence lines were audited by hand** on the matches. A first version was right for the wrong reason about 30% of the time (job titles such as `Complete job name: govulncheck`, install lines such as `ruff==0.14.10`, compiler flags, echoed script text). The final rules ignore command, env and title lines; about 9% of remaining matches still cite a line that is not proof.
- **The learned model did worse than nothing on real logs:** trained on synthetic logs only, it scored 39.6% (below always guessing `real_bug`), and 54.7% in the rules-then-model pipeline with the original rules. It was not retrained on real logs.
- **Flaky tests are the weak spot**: 3 of 53 labelled logs, 1 caught. A single log cannot show whether a rerun passes, and real flaky failures look like real failures.
- Caveats: one annotator, no inter-rater check, 53 classifiable logs (a 95% interval on 87% is about ±9 points), a failed-run sample skewed to big projects, and "other" calls are my judgement.

Reproduce (needs `gh` logged in; logs older than ~90 days are gone):
```bash
python scripts/collect_real_logs.py --out data/real_logs --per-repo 4   # ~25 min; re-fetches by repo, so job ids differ from the labelled set
python scripts/eval_real.py data/real_logs --old-rules data/rules_v1.json   # only meaningful on the labelled job ids
```

## Pipeline (`citriage/triage.py`)

1. **Rules** (`rules.json`, 18 signatures): regex families per class, ignoring benign lines (warnings, passing tests). A class's score is its best rule weight plus a small bonus for corroborating rules. If one class owns at least 80% of the score, the rules decide, and the matched lines are the evidence.
2. **ML** (`ml.py`): TF-IDF + logistic regression on the *error window* only (failure-looking lines plus neighbours, numbers masked), because ~99% of a CI log is noise. Used when rules are ambiguous and the model is at least 60% sure.
3. **Gemini** (`llm.py`): for runs neither stage can decide. Sends only the extracted window, never the whole log, and rejects outputs with an invalid label. Implemented and unit-tested against a fake transport; **not run live** (no API key at build time).

## Synthetic results (1,200 held-out synthetic runs per row)

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

### Next steps

More real labels from more than one annotator; retrain the model on real logs; try the Gemini fallback on the `unsure` runs (still not run live, no API key).
