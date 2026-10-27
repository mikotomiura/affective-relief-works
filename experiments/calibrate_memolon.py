"""Fix the map from the Japanese MEmoLon scale to [-1, 1] by a fit of English MEmoLon to NRC VAD v2.1.

Usage:  uv run python experiments/calibrate_memolon.py     (compute and write the result; refuses to overwrite)
Conditions: experiments/conditions/calibrate_memolon.json (committed before this script was run)
Output: results/calibrate_memolon.json
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
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


def load(entry: dict) -> dict[str, tuple[float, float]]:
    """Read the file once, check the SHA-256 of those bytes against the conditions, then parse the same bytes."""
    p = ROOT / entry["file"].split(" ")[0]
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise SystemExit(f"{p} does not match the SHA-256 of the conditions")
    rows = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), delimiter="\t", quoting=csv.QUOTE_NONE)
    head = next(rows)
    if head[0] not in ("term", "word") or head[1:3] != list(DIMS):
        raise SystemExit(f"unexpected header in {p}: {head}")
    out = {}
    for i, r in enumerate(rows, 2):
        if len(r) < 3 or not r[0]:
            raise SystemExit(f"line {i} of {p} has too few fields or an empty word")
        if r[0] in out:
            raise SystemExit(f"duplicate entry {r[0]!r} in {p}")
        v, a = float(r[1]), float(r[2])
        if not (math.isfinite(v) and math.isfinite(a)):
            raise SystemExit(f"non-finite value on line {i} of {p}")
        out[r[0]] = (v, a)
    return out


def percentiles(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    n = len(xs)
    return {str(q): xs[round(q / 100 * (n - 1))] for q in PCTS}


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    commit = run("rev-parse", "HEAD")
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise SystemExit(f"unexpected commit from git: {commit!r}")
    paths = ["experiments", "pyproject.toml", "uv.lock", ".python-version"]
    return {"git_commit": commit, "git_dirty": bool(run("status", "--porcelain", "--", *paths))}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    cond_raw = COND.read_bytes()  # read once: the conditions used and the hash recorded come from the same bytes
    c = json.loads(cond_raw)
    if platform.python_version() != c["python"]:
        raise SystemExit(f"Python is {platform.python_version()}, not {c['python']} as in the conditions")
    state = git_state()
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    lx = c["lexicons"]
    nrc, en, ja = (load(lx[k]) for k in ("nrc_vad", "memolon_en", "memolon_ja"))
    common = sorted(w for w in nrc if w in en)
    res: dict = {"nrc_entries": len(nrc), "english_memolon_entries": len(en), "common_words": len(common),
                 "japanese_entries": len(ja), "percentile_method": c["percentiles"], "dims": {}}
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
                          "share_clipped_low": sum(m < -1.0 for m in mapped) / len(mapped),
                          "share_clipped_high": sum(m > 1.0 for m in mapped) / len(mapped),
                          "zero_point_on_memolon_scale": -a / b if b != 0 else None,
                          "nrc_common_percentiles": percentiles(y)}
    out = {"meta": {"conditions_sha256": hashlib.sha256(cond_raw).hexdigest(), "script_sha256": sha256(SCRIPT),
                    "lexicon_sha256_checked": {k: v["sha256"] for k, v in lx.items()},
                    "uv_lock_sha256": sha256(ROOT / "uv.lock"), "pyproject_sha256": sha256(ROOT / "pyproject.toml"),
                    "python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                    **state,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "result": res}
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("x") as f:  # exclusive creation: never overwrites a result written in the meantime
        f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
