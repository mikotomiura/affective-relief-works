"""Fix the map from the Japanese MEmoLon scale to [-1, 1] by a fit of English MEmoLon to NRC VAD v2.1.

Usage:  uv run python experiments/calibrate_memolon.py     (compute and write the result; refuses to overwrite)
Conditions: experiments/conditions/calibrate_memolon.json (committed before this script was run)
Output: results/calibrate_memolon.json
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
COND = ROOT / "experiments" / "conditions" / "calibrate_memolon.json"
OUT = ROOT / "results" / "calibrate_memolon.json"
DIMS = ("valence", "arousal")
PCTS = (0, 1, 25, 50, 75, 99, 100)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def path_of(entry: dict) -> Path:
    p = ROOT / entry["file"].split(" ")[0]
    if sha256(p) != entry["sha256"]:
        raise SystemExit(f"{p} does not match the SHA-256 of the conditions")
    return p


def load(p: Path) -> dict[str, tuple[float, float]]:
    with p.open(encoding="utf-8", newline="") as f:
        rows = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        head = next(rows)
        if head[1:3] != list(DIMS):
            raise SystemExit(f"unexpected header in {p}: {head}")
        out = {}
        for r in rows:
            if r[0] in out:
                raise SystemExit(f"duplicate entry {r[0]!r} in {p}")
            out[r[0]] = (float(r[1]), float(r[2]))
        return out


def percentiles(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    n = len(xs)
    return {str(q): xs[round(q / 100 * (n - 1))] for q in PCTS}


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain", "--", "experiments", "pyproject.toml", "uv.lock"))}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    c = json.loads(COND.read_text())
    state = git_state()
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    lx = c["lexicons"]
    nrc, en, ja = (load(path_of(lx[k])) for k in ("nrc_vad", "memolon_en", "memolon_ja"))
    common = sorted(w for w in nrc if w in en)
    res: dict = {"common_words": len(common), "japanese_entries": len(ja), "dims": {}}
    for j, d in enumerate(DIMS):
        x = [en[w][j] for w in common]
        y = [nrc[w][j] for w in common]
        b, a = statistics.linear_regression(x, y)
        mapped = [b * v[j] + a for v in ja.values()]
        clipped = [min(1.0, max(-1.0, m)) for m in mapped]
        res["dims"][d] = {"b": b, "a": a, "r": statistics.correlation(x, y),
                          "japanese_mapped_percentiles": percentiles(mapped),
                          "japanese_mapped_clipped_percentiles": percentiles(clipped),
                          "share_changed_by_clip": sum(m != q for m, q in zip(mapped, clipped)) / len(mapped),
                          "nrc_common_percentiles": percentiles(y)}
    out = {"meta": {"conditions_sha256": sha256(COND), "script_sha256": sha256(SCRIPT),
                    "lexicon_sha256": {k: v["sha256"] for k, v in lx.items()},
                    "python": platform.python_version(), **state,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "result": res}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
