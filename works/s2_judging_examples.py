"""Five images of the blind judgement for supplement S2 (S2.9), copied unchanged from the judging spreadsheet.

Usage:  uv run python works/s2_judging_examples.py   (refuses to overwrite)
Conditions: works/conditions/s2_judging_examples.json (committed before this script was written)
Input:  advice/blind-judgement/sealed/key.json and advice/blind-judgement/judgement.xlsx (outside the repository)
Output: results/s2_judging_examples/
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "works"))

import _common as cm  # noqa: E402

COND = ROOT / "works" / "conditions" / "s2_judging_examples.json"
OUT_DIR = ROOT / "results" / "s2_judging_examples"
VERSIONS = ("G0", "P", "SPC")
CONTROLS = ("set_differs", "same_image")
ROW = "words20"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    cm.same_as_head([COND, Path(__file__).resolve()])
    key_path, xlsx_path = ROOT / c["inputs"]["key"], ROOT / c["inputs"]["spreadsheet"]
    key = json.loads(key_path.read_bytes())
    xlsx = xlsx_path.read_bytes()
    if sha(xlsx) != key["spreadsheet_sha256"]:
        raise SystemExit("the spreadsheet is not the one of the key")
    items = key["items"]
    chosen = []
    for v in VERSIONS:
        cand = [i for i in items if i["kind"] == "test" and i["top"]["version"] == v and i["bottom"]["version"] == v
                and i["top"]["row"] == ROW and i["bottom"]["row"] == ROW]
        if not cand:
            raise SystemExit(f"no test item of {v} on {ROW}")
        chosen.append(min(cand, key=lambda i: i["number"]))
    for k in CONTROLS:
        cand = [i for i in items if i["kind"] == k]
        if not cand:
            raise SystemExit(f"no item of kind {k}")
        chosen.append(min(cand, key=lambda i: i["number"]))
    with zipfile.ZipFile(xlsx_path) as z:
        media = {n: z.read(n) for n in z.namelist() if n.startswith("xl/media/")}
    by_sha: dict[str, list[str]] = {}
    for n, b in media.items():
        by_sha.setdefault(sha(b), []).append(n)
    tmp = OUT_DIR.with_name(OUT_DIR.name + ".tmp")
    if tmp.exists():
        raise SystemExit(f"{tmp} exists")
    tmp.mkdir(parents=True)
    rec = []
    for it in chosen:
        names = by_sha.get(it["image_sha256"], [])
        if not names:
            raise SystemExit(f"item {it['number']}: no image of the spreadsheet has its SHA-256")
        data = media[names[0]]
        (tmp / f"item{it['number']}.png").write_bytes(data)
        if sha((tmp / f"item{it['number']}.png").read_bytes()) != it["image_sha256"]:
            raise SystemExit(f"item {it['number']}: the copy differs")
        rec.append({"number": it["number"], "kind": it["kind"], "top": it["top"], "bottom": it["bottom"],
                    "media": names, "sha256": it["image_sha256"]})
    record = {"conditions_sha256": sha(raw), "script_sha256": cm.sha256(Path(__file__).resolve()),
              "key_sha256": sha(key_path.read_bytes()), "spreadsheet_sha256": sha(xlsx), "items": rec}
    (tmp / "s2_judging_examples.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")
    tmp.rename(OUT_DIR)
    print("\n".join(f"item{r['number']}: {r['kind']} {r['top']['version']} {r['top']['row']}" for r in rec))


if __name__ == "__main__":
    main()
