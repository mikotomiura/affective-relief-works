"""Check how the Japanese text step responds to the map of the lexicon scale, the order of clip and mean and the word
selection, on check sentences written for this purpose (no text of any work).

Usage:  uv run python checks/japanese_map_sensitivity.py      (compute and write the result; refuses to overwrite)
Conditions: checks/conditions/japanese_map_sensitivity.json (committed before this script was run)
Input:  checks/reference/japanese_check_sentences.tsv, results/calibrate_memolon.json and the Japanese MEmoLon
        lexicon (external/, not in the repository)
Output: results/japanese_map_sensitivity.json
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import hashlib
import io
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# compile the imported modules afresh: a cached .pyc in src/__pycache__ (ignored by git) is never used, so the code
# that runs is the source compared with HEAD below
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))

import atact  # noqa: E402
from atact.mapping import params  # noqa: E402
from japanese_text import Lexicon, LinearMap, Tagger, clip, load_memolon, sentence_point  # noqa: E402

SCRIPT = Path(__file__).resolve()
COND = ROOT / "checks" / "conditions" / "japanese_map_sensitivity.json"
OUT = ROOT / "results" / "japanese_map_sensitivity.json"
MODULE = ROOT / "src" / "japanese_text.py"
PCTS = (0, 25, 50, 75, 100)
DIMS = ("valence", "arousal")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    commit = run("rev-parse", "HEAD")
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise SystemExit(f"unexpected commit from git: {commit!r}")
    paths = ["checks", "src", "results/calibrate_memolon.json", "pyproject.toml", "uv.lock", ".python-version"]
    return {"git_commit": commit, "git_dirty": bool(run("status", "--porcelain", "--", *paths))}


def same_as_head(paths: list[Path]) -> None:
    """The files that are run or read must be the committed bytes (the status check alone would pass an ignored file)."""
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        head = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
        if head != p.read_bytes():
            raise SystemExit(f"{rel} differs from its committed version")


def percentiles(xs: list[float]) -> dict[str, float] | None:
    """Nearest index, as in experiments/calibrate_memolon.py."""
    if not xs:
        return None
    xs = sorted(xs)
    return {str(q): xs[round(q / 100 * (len(xs) - 1))] for q in PCTS}


@dataclass(frozen=True)
class RankMap:
    """Map C: per dimension, 2 F(x) - 1 with the mid-rank distribution function F of the lexicon values, so that the
    median of the lexicon goes to 0 and the range to (-1, 1)."""

    sorted_v: list[float]
    sorted_a: list[float]

    @staticmethod
    def _c(xs: list[float], x: float) -> float:
        lo, hi = bisect.bisect_left(xs, x), bisect.bisect_right(xs, x)
        return (lo + hi) / len(xs) - 1.0

    def __call__(self, v: float, a: float) -> tuple[float, float]:
        return self._c(self.sorted_v, v), self._c(self.sorted_a, a)


def check_sentences(c: dict) -> list[dict]:
    raw = (ROOT / c["sentences"]["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != c["sentences"]["sha256"]:
        raise SystemExit("the check sentences differ from those named in the conditions")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""), delimiter="\t",
                               quoting=csv.QUOTE_NONE))
    if [r["id"] for r in rows] != c["sentences"]["ids"]:
        raise SystemExit("the ids of the check sentences differ from those of the conditions")
    return rows


def calibration(c: dict) -> LinearMap:
    raw = (ROOT / c["calibration"]["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != c["calibration"]["sha256"]:
        raise SystemExit("the calibration result differs from the one named in the conditions")
    d = json.loads(raw)["result"]["dims"]
    return LinearMap(d["valence"]["b"], d["valence"]["a"], d["arousal"]["b"], d["arousal"]["a"])


def versions() -> dict:
    d = json.loads(metadata.distribution("atact").read_text("direct_url.json"))
    return {"python": platform.python_version(), "fugashi": metadata.version("fugashi"),
            "ipadic": metadata.version("ipadic"), "numpy": metadata.version("numpy"),
            "atact_commit": d["vcs_info"]["commit_id"]}


def run_combo(rows: list[dict], lex: Lexicon, tagger: Tagger, vmap, select: str, order: str, linear: LinearMap | None
              ) -> dict:
    """The clip is counted where it acts: over matched word occurrences for order "word", over matched sentences (on
    the mapped mean) for order "sentence"."""
    pts = []
    units = 0
    clipped = {d: 0 for d in DIMS}
    for r in rows:
        p = sentence_point(r["sentence"], lex, tagger, vmap, select, order)
        f, s, g = (float(x) for x in params(p.V, p.A))
        pts.append({"id": r["id"], "Z": p.Z, "n_candidates": p.n_candidates, "unmatched": p.unmatched,
                    "matched": p.matched, "V": p.V, "A": p.A, "f": f, "s": s, "gamma": g})
        if linear is None or p.unmatched:
            continue
        if order == "word":
            raws = [linear.raw(*lex.entries[w]) for w in p.matched]
        else:
            raws = [linear.raw(sum(lex.entries[w][0] for w in p.matched) / p.Z,
                               sum(lex.entries[w][1] for w in p.matched) / p.Z)]
        for x, y in raws:
            units += 1
            clipped["valence"] += x != clip(x)
            clipped["arousal"] += y != clip(y)
    hit = [q for q in pts if not q["unmatched"]]
    summary = {"sentences": len(pts), "unmatched": len(pts) - len(hit),
               "share_unmatched": (len(pts) - len(hit)) / len(pts),
               "matched_occurrences": sum(q["Z"] for q in pts),
               "candidate_occurrences": sum(q["n_candidates"] for q in pts),
               "percentiles_over_matched_sentences": {k: percentiles([q[k] for q in hit])
                                                      for k in ("V", "A", "f", "s", "gamma")}}
    if linear is not None:
        unit = "matched_word_occurrences" if order == "word" else "matched_sentences"
        summary[f"share_of_{unit}_changed_by_clip"] = {d: clipped[d] / units if units else None for d in DIMS}
    return {"summary": summary, "sentences": pts}


def pair_differences(pts: list[dict], pairs: list[list[str]]) -> dict:
    """b - a within each pair; null when either sentence is unmatched ((0, 0) is a flag, not a value)."""
    by = {q["id"]: q for q in pts}
    out = {}
    for a, b in pairs:
        both = not (by[a]["unmatched"] or by[b]["unmatched"])
        out[f"{a}-{b}"] = {k: (by[b][k] - by[a][k]) if both else None for k in ("V", "A", "f", "s", "gamma")} | {
            "Z_a": by[a]["Z"], "Z_b": by[b]["Z"], "matched_a": by[a]["matched"], "matched_b": by[b]["matched"]}
    return out


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    cond_raw = COND.read_bytes()  # read once: the conditions used and the hash recorded come from the same bytes
    c = json.loads(cond_raw)
    got = versions()
    for k, v in got.items():
        if v != c["versions"][k]:
            raise SystemExit(f"{k} is {v}, not {c['versions'][k]} as in the conditions")
    state = git_state()
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    same_as_head([SCRIPT, MODULE, COND, ROOT / c["sentences"]["file"], ROOT / c["calibration"]["file"]])
    atact_dir = Path(atact.__file__).parent
    atact_sha = {n: sha256(atact_dir / n) for n in ("__init__.py", "mapping.py")}
    if atact_sha != c["atact_sha256"]:
        raise SystemExit("the installed atact files differ from those named in the conditions")
    rows = check_sentences(c)
    ids = {r["id"] for r in rows}
    if any(i not in ids for pair in c["pairs"] for i in pair) or c["main_combination"] != "B_calibrated/word/pos":
        raise SystemExit("the pairs or the main combination do not match the check sentences and the step")
    b_map = calibration(c)
    lex = load_memolon(ROOT / c["lexicon"]["path"])
    if lex.sha256 != c["lexicon"]["sha256"]:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    tagger = Tagger()
    if tagger.dicdir_sha256 != c["ipadic_dicdir_sha256"]:
        raise SystemExit("the IPAdic dictionary files differ from those named in the conditions")
    a_map = LinearMap(0.25, -1.25, 0.25, -1.25)
    c_map = RankMap(sorted(v for v, _ in lex.entries.values()), sorted(a for _, a in lex.entries.values()))
    maps = {"A_nominal": (a_map, a_map), "B_calibrated": (b_map, b_map), "C_rank": (c_map, None)}
    combos = {}
    for mname, (vmap, linear) in maps.items():
        for order in ("word", "sentence"):
            for select in ("pos", "all"):
                combos[f"{mname}/{order}/{select}"] = run_combo(rows, lex, tagger, vmap, select, order, linear)
    main_key = c["main_combination"]
    res = {"combinations": combos,
           "pairs_in_main_combination": pair_differences(combos[main_key]["sentences"], c["pairs"]),
           "nominal_neutral_5_under_B": dict(zip(("V", "A"), b_map.raw(5.0, 5.0))),
           "lexicon_entries": len(lex.entries)}
    out = {"meta": {"conditions_sha256": hashlib.sha256(cond_raw).hexdigest(), "script_sha256": sha256(SCRIPT),
                    "module_sha256": sha256(MODULE), "lexicon_sha256_checked": lex.sha256,
                    "ipadic_dicdir_sha256": tagger.dicdir_sha256, "atact_sha256": atact_sha,
                    "versions": got,
                    "uv_lock_sha256": sha256(ROOT / "uv.lock"), "pyproject_sha256": sha256(ROOT / "pyproject.toml"),
                    "platform": f"{platform.system()} {platform.machine()}", **state,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "result": res}
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("x") as f:  # exclusive creation: never overwrites a result written in the meantime
        f.write(json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    for k, v in combos.items():
        print(k, json.dumps(v["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
