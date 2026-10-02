"""Synthetic GitHub Actions failure logs with known labels and a planted proof line.

NOT GHALogs. Logs carry GitHub's timestamp prefix, step groups and build noise. Each class has several real-world
signature families; `signal` is the chance a run contains a strong signature at all (otherwise only a generic
"exit code 1" line, which is undecidable), and `confuse` the chance a decoy from another class appears as a
non-fatal line. Accuracy measured here reflects this generator, not the wild.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

SIGNALS = {
    "flaky_test": [
        "FAIL src/api/{m}.test.ts\n  ● {m} › handles request\n    thrown: \"Exceeded timeout of 5000 ms for a test.\"",
        "Error: listen EADDRINUSE: address already in use :::{p}",
        "Error: read ECONNRESET\n    at TCP.onStreamRead (node:internal/stream_base_commons:217:20)",
        "tests/test_{m}.py::test_sync RERUN [100%]\ntests/test_{m}.py::test_sync FAILED",
        "TimeoutError: Waiting for selector `#submit` failed: Waiting failed: 30000ms exceeded",
        "Retrying 2/3 for test {m}_integration\nAttempt 2 of 3 failed",
        "WebDriverException: Message: element not interactable; Element is not clickable at point ({p}, 412)",
    ],
    "dependency": [
        "npm ERR! code E404\nnpm ERR! 404 Not Found - GET https://registry.npmjs.org/@{m}/core - Not found",
        "npm ERR! code ERESOLVE\nnpm ERR! ERESOLVE unable to resolve dependency tree",
        "ERROR: Could not find a version that satisfies the requirement {m}==9.{p}\nERROR: No matching distribution found for {m}==9.{p}",
        "[ERROR] Failed to execute goal on project app: Could not resolve dependencies for project com.acme:app:jar:1.{p}",
        "go: github.com/acme/{m}@v1.{p}.0: reading github.com/acme/{m}/go.mod at revision v1.{p}.0: 404 Not Found",
        "error: failed to select a version for the requirement `{m} = \"^0.{p}\"`",
        "npm ERR! `npm ci` can only install packages when your package.json and package-lock.json are in sync",
    ],
    "infrastructure": [
        "##[error]The runner has received a shutdown signal. This can happen when the runner service is stopped.",
        "System.IO.IOException: No space left on device : '/home/runner/work/_temp/{p}'",
        "fatal: unable to access 'https://github.com/acme/{m}.git/': Could not resolve host: github.com",
        "Error response from daemon: toomanyrequests: You have reached your pull rate limit.",
        "##[error]The hosted runner lost communication with the server. Anything in your workflow that terminates the runner process will cause this.",
        "curl: (22) The requested URL returned error: 503 Service Unavailable",
        "Error: The operation was canceled.\n##[error]Process completed with exit code 143.",
    ],
    "real_bug": [
        "src/{m}.ts({p},12): error TS2322: Type 'string' is not assignable to type 'number'.",
        "FAILED tests/test_{m}.py::test_total - AssertionError: assert {p} == {p2}",
        "error[E0308]: mismatched types\n  --> src/{m}.rs:{p}:14",
        "Traceback (most recent call last):\n  File \"app/{m}.py\", line {p}, in run\nNameError: name '{m}_conf' is not defined",
        "{m}.go:{p}:2: undefined: {m}Handler\n--- FAIL: Test{m} (0.00s)",
        "java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"s\" is null\n\tat com.acme.{m}.Parser.parse(Parser.java:{p})",
        "  12 problems (3 errors, 9 warnings)\n##[error]eslint found errors in src/{m}.js",
    ],
}
NOISE = [
    "##[group]Run actions/checkout@v4", "Syncing repository: acme/{m}", "##[endgroup]", "added {p} packages in {p2}s",
    "npm WARN deprecated {m}@1.{p}: no longer supported", "PASS src/{m}/{m}.test.ts ({p2} ms)",
    "tests/test_{m}.py::test_ok PASSED [ {p}%]", "Downloading {m}-1.{p}.tar.gz (23 kB)", "retrying download (1/3)",
    "Building wheel for {m} (setup.py): finished with status 'done'", "::debug::Evaluating condition for step '{m}'",
    "  CC  obj/{m}_{p}.o", "[INFO] --- maven-compiler-plugin:3.11.0:compile (default-compile) @ app ---",
    "Test Suites: {p2} passed, {p2} total", "ok  \tgithub.com/acme/{m}\t0.{p}s", "skipping {m} (up to date)",
    "test_timeout_handler PASSED", "0 failed, {p} passed",
]
WORDS = ["auth", "cache", "router", "billing", "queue", "parser", "worker", "client", "search", "upload", "session", "export"]
CLASS_P = {"flaky_test": 0.2, "dependency": 0.25, "infrastructure": 0.2, "real_bug": 0.35}


def _fmt(t: str, rng: random.Random) -> str:
    return t.format(m=rng.choice(WORDS), p=rng.randint(2, 999), p2=rng.randint(2, 99))


def generate(n: int = 3000, seed: int = 5, signal: float = 0.82, confuse: float = 0.10, families: str = "all"):
    """families: "all", "seen" (every family except the last of each class) or "novel" (only the last of each class)."""
    rng = random.Random(seed)
    labels = list(CLASS_P)
    out = []
    t0 = datetime(2025, 3, 1, 12, 0, 0)
    for rid in range(n):
        label = rng.choices(labels, weights=[CLASS_P[l] for l in labels])[0]
        body = [_fmt(rng.choice(NOISE), rng) for _ in range(rng.randint(60, 320))]
        proof = None
        if rng.random() < signal:
            pool = SIGNALS[label]
            pool = pool[:-1] if families == "seen" else pool[-1:] if families == "novel" else pool
            sig = _fmt(rng.choice(pool), rng)
            pos = rng.randint(len(body) // 2, len(body))  # failures happen late
            body[pos:pos] = sig.split("\n")
            proof = list(range(pos, pos + len(sig.split("\n"))))
        if rng.random() < confuse:  # decoy: other class's text as a harmless line
            other = rng.choice([l for l in labels if l != label])
            decoy = _fmt(rng.choice(SIGNALS[other]), rng).split("\n")[0]
            body.insert(rng.randint(0, len(body) // 2), "npm WARN " + decoy)
            if proof:  # inserted before proof shifts it
                pass
        body.append("##[error]Process completed with exit code 1.")
        # recompute proof indices after possible decoy insertion by searching the signature's first line
        if proof:
            first = sig.split("\n")[0]
            proof = [i for i, l in enumerate(body) if l == first or l in sig.split("\n")]
        ts = t0 + timedelta(seconds=rid * 37)
        lines = [f"{(ts + timedelta(milliseconds=i * 40)).strftime('%Y-%m-%dT%H:%M:%S.%f')}0Z {l}" for i, l in enumerate(body)]
        out.append({"run_id": rid, "label": label, "lines": lines, "proof": proof or []})
    return out
