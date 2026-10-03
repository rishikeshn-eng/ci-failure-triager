"""Collect failed GitHub Actions job logs from public repos with your `gh` login (read-only API calls).

    python scripts/collect_real_logs.py --out data/real_logs --per-repo 5

Takes recent failed runs, then the first failed job of each run, and saves that job's log text (capped at 2 MB).
GitHub keeps logs ~90 days, so a labelled set built from run/job ids can only be re-fetched while they last.
"""
import argparse, json, os, re, subprocess, sys

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

REPOS = ["home-assistant/core", "microsoft/vscode", "facebook/react", "rust-lang/rust", "nodejs/node", "numpy/numpy",
         "pandas-dev/pandas", "apache/airflow", "django/django", "angular/angular", "vercel/next.js", "denoland/deno",
         "grafana/grafana", "golang/go", "kubernetes/kubernetes", "pytorch/pytorch", "huggingface/transformers",
         "scikit-learn/scikit-learn", "tokio-rs/tokio", "prometheus/prometheus", "hashicorp/terraform", "electron/electron",
         "godotengine/godot", "ansible/ansible", "pallets/flask", "psf/requests", "microsoft/TypeScript", "vuejs/core"]


def gh(path, raw=False):
    try:
        r = subprocess.run(["gh", "api", "--allow-escape-sequences", path], capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode:
        return None
    return ANSI.sub("", r.stdout) if raw else json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_logs")
    ap.add_argument("--per-repo", type=int, default=5)
    ap.add_argument("--repos", nargs="*", default=REPOS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = []
    for repo in a.repos:
        runs = gh(f"repos/{repo}/actions/runs?status=failure&per_page=30") or {}
        got = 0
        seen_wf = set()
        for run in runs.get("workflow_runs", []):
            if got >= a.per_repo:
                break
            if run["name"] in seen_wf and got < a.per_repo - 1:
                continue  # prefer variety of workflows
            jobs = (gh(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=50") or {}).get("jobs", [])
            bad = [j for j in jobs if j["conclusion"] == "failure"]
            if not bad:
                continue
            j = bad[0]
            log = gh(f"repos/{repo}/actions/jobs/{j['id']}/logs", raw=True)
            if not log or len(log) > 2_000_000 or len(log.splitlines()) < 5:
                continue
            seen_wf.add(run["name"])
            fn = f"{repo.replace('/', '__')}__{j['id']}.log"
            open(os.path.join(a.out, fn), "w").write(log)
            meta.append({"file": fn, "repo": repo, "run_id": run["id"], "job_id": j["id"], "workflow": run["name"],
                         "job": j["name"], "event": run["event"], "lines": len(log.splitlines())})
            got += 1
        print(repo, got, flush=True)
    json.dump(meta, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(len(meta), "logs")


if __name__ == "__main__":
    main()
