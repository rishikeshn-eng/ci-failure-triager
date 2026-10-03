"""Inject results.json (+ real_results.json) into the single-file UI -> docs/index.html."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
data = json.loads((root / "docs" / "results.json").read_text())
real = root / "docs" / "real_results.json"
if real.exists():
    data["real"] = json.loads(real.read_text())
html = (root / "site" / "template.html").read_text().replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
(root / "docs" / "index.html").write_text(html)
print("wrote docs/index.html", len(html) // 1024, "KiB")
