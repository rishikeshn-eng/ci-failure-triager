"""Optional Gemini escalation for runs the cheap stages cannot decide. Sends only the extracted evidence window."""
from __future__ import annotations

import json
import os
import urllib.request

from .rules import LABELS

PROMPT = """You triage failed CI runs. Classify the failure as exactly one of:
flaky_test (passes on rerun: timeouts, races, port clashes, retries),
dependency (package/module could not be resolved, fetched or locked),
infrastructure (runner, disk, network, rate limits, cancelled/killed),
real_bug (the code under test is wrong: compile, assertion, exception, lint).
Reply with JSON only: {{"label": "...", "evidence_line": <index of the line that proves it>}}.

Lines (index: text):
{lines}
"""


class GeminiTriage:
    def __init__(self, model=None, api_key=None, transport=None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.key and transport is None:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.transport = transport or self._http
        self.calls = self.tokens_in = self.tokens_out = self.rejected = 0

    def _http(self, body):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    def classify(self, window_lines: list[str]) -> dict | None:
        prompt = PROMPT.format(lines="\n".join(f"{i}: {l[:300]}" for i, l in enumerate(window_lines)))
        resp = self.transport({"contents": [{"parts": [{"text": prompt}]}],
                               "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        self.calls += 1
        meta = resp.get("usageMetadata") or {}
        self.tokens_in += meta.get("promptTokenCount", len(prompt) // 4)
        try:
            out = json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
            assert out["label"] in LABELS
            idx = int(out.get("evidence_line", -1))
        except Exception:
            self.rejected += 1
            return None
        self.tokens_out += meta.get("candidatesTokenCount", 20)
        return {"label": out["label"], "evidence": window_lines[idx] if 0 <= idx < len(window_lines) else None}
