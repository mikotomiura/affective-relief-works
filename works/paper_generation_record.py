"""The record of the quantities of generation of figure 6 of the paper: V, A, f, s and gamma of the 20 words of the two
trial authors under map B (version G0) and under the author-equal percentiles (version P), with the fit of map B and the
Jacobian of the rule.

Usage:  uv run python works/paper_generation_record.py   (refuses to overwrite)
Conditions: works/conditions/paper_generation_record.json (committed before this script was run)
Input:  the committed records named in the conditions (no text, no lexicon, no drawing)
Output: results/paper_generation_record/
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from atact.mapping import jacobian, params  # noqa: E402

import _common as cm  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_generation_record.json"
OUT_DIR = ROOT / "results" / "paper_generation_record"
CODE = ["works/paper_generation_record.py", "works/_common.py"]
N_AUTHORS, N_WORDS, VERSIONS = 2, 20, ("G0", "P")
QUANTITIES = ("V", "A", "f", "s", "gamma")
RECORD = "generation_record.json"


def va(version: str, v: dict) -> tuple[float, float]:
    """V and A of a word in a version, the same formula as va of works/paper_figure6_scale.py."""
    if version == "G0":
        return v["V"], v["A"]
    return 2 * v["p_valence_author_equal"] - 1, 2 * v["p_arousal_author_equal"] - 1


def summary(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    q1, med, q3 = (float(x) for x in np.quantile(a, [0.25, 0.5, 0.75], method="linear"))
    return {"min": float(a.min()), "q1": q1, "median": med, "q3": q3, "max": float(a.max())}


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    inp = c["inputs"]
    path = {k: ROOT / v["path"] for k, v in inp.items() if isinstance(v, dict)}
    cm.check_file(path["values"], inp["values"]["sha256"], "values")
    cm.same_as_head([COND, *path.values(), *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "results/sketch_sp_color", "results/work2_words", "results/paper_figure6_scale",
                          "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    if not c["outputs"]["record"].startswith(RECORD + ":"):
        raise SystemExit("the record of the conditions is not named " + RECORD)
    k = c["constants"]
    if (k["authors"], k["words_per_author"], tuple(k["versions"])) != (N_AUTHORS, N_WORDS, VERSIONS):
        raise SystemExit("the constants of the conditions differ from those of the script")
    words = json.loads(path["words"].read_bytes())
    wc = json.loads(path["words_conditions"].read_bytes())
    jc = json.loads(path["judgement_conditions"].read_bytes())
    f6c = json.loads(path["figure6_conditions"].read_bytes())
    f6 = json.loads(path["figure6_record"].read_bytes())
    cal = json.loads(path["calibration"].read_bytes())
    if words["meta"]["conditions_sha256"] != cm.sha256(path["words_conditions"]) or words["meta"].get("git_dirty") is not False:
        raise SystemExit("words.json was not computed on the conditions named, on a clean tree")
    if (f6["meta"]["conditions_sha256"] != cm.sha256(path["figure6_conditions"]) or f6["meta"].get("git_dirty") is not False
            or f6["meta"]["inputs_sha256"]["values"] != inp["values"]["sha256"]):
        raise SystemExit("the record of figure 6 was not made on the conditions and values named, on a clean tree")
    if (f6c["inputs"]["values"]["sha256"] != inp["values"]["sha256"]
            or jc["inputs"]["values_sha256"] != inp["values"]["sha256"]):
        raise SystemExit("values.json is not the one of figure 6 and of the judgement")
    w2 = json.loads(path["work2_conditions"].read_bytes())
    if words["meta"]["work2_conditions_sha256"] != cm.sha256(path["work2_conditions"]):
        raise SystemExit("work2.json is not the one named in words.json")
    if cal["meta"].get("git_dirty") is not False or cm.sha256(path["calibration"]) != w2["map"]["sha256"]:
        raise SystemExit("calibrate_memolon.json is not the map of work2.json computed on a clean tree")
    cm.check_generator(wc["generator"])
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # C1: the constants ---------------------------------------------------
    authors = wc["trial_authors"]
    if len(authors) != N_AUTHORS or authors != jc["authors"] or list(f6["words"]) != authors:
        raise SystemExit(f"C1: the trial authors differ or are not {N_AUTHORS}: {authors}")
    seq = {a: [w["word"] for w in words["authors"][a]["words"]] for a in authors}
    if any(len(seq[a]) != N_WORDS or seq[a] != f6["words"][a] for a in authors):
        raise SystemExit(f"C1: an author has not the same {N_WORDS} words in words.json and in the record of figure 6")

    values = json.loads(path["values"].read_bytes())["values"]
    rows: dict = {v: {a: [] for a in authors} for v in VERSIONS}
    c2 = c3 = 0
    for a in authors:
        for w in words["authors"][a]["words"]:
            entry = values.get(w["word"])
            if entry is None:
                raise SystemExit(f"{w['word']} is not in values.json")
            for version in VERSIONS:
                if version == "P":  # C3
                    for key in ("p_valence_author_equal", "p_arousal_author_equal"):
                        if not 0.0 <= entry[key] <= 1.0:
                            raise SystemExit(f"C3: {w['word']}: {key} = {entry[key]} outside [0, 1]")
                V, A = va(version, entry)
                if not (-1.0 <= V <= 1.0 and -1.0 <= A <= 1.0):  # C3: params would clip and the record would mislead
                    raise SystemExit(f"C3: {version} {w['word']}: V = {V}, A = {A} outside [-1, 1]")
                c3 += 1
                f, s, g = (float(x) for x in params(V, A))
                if version == "G0":
                    if (V, A) != (w["V"], w["A"]) or (f, s, g) != (w["f"], w["s"], w["gamma"]):
                        raise SystemExit(f"C2: {w['word']}: V, A, f, s or gamma differ from words.json")
                    c2 += 1
                rows[version][a].append({"word": w["word"], "V": V, "A": A, "f": f, "s": s, "gamma": g})
    n = sum(len(rows[v][a]) for v in VERSIONS for a in authors)
    if n != len(VERSIONS) * N_AUTHORS * N_WORDS or c3 != n or c2 != N_AUTHORS * N_WORDS:
        raise SystemExit(f"recorded {n} rows, C2 {c2}, C3 {c3}; not {len(VERSIONS) * N_AUTHORS * N_WORDS} and "
                         f"{N_AUTHORS * N_WORDS}")

    # C4: the Jacobian ----------------------------------------------------
    J = jacobian()  # rows f, s, gamma; columns d/dV, d/dA
    if J.shape != (3, 2) or J[0, 0] != 0.0 or J[1, 1] != 0.0 or 0.0 in (J[0, 1], J[1, 0], J[2, 0], J[2, 1]):
        raise SystemExit(f"C4: the Jacobian is not of the expected form: {J.tolist()}")
    base = np.array(params(0.0, 0.0), dtype=float)
    fd = np.column_stack([(np.array(params(0.5, 0.0), dtype=float) - base) / 0.5,
                          (np.array(params(0.0, 0.5), dtype=float) - base) / 0.5])
    if not np.allclose(fd, J, rtol=0.0, atol=1e-12):
        raise SystemExit(f"C4: params does not have the Jacobian of jacobian(): {fd.tolist()}")

    # C5: the fit of map B, copied ----------------------------------------
    fit = {d: {key: cal["result"]["dims"][d][key] for key in ("b", "a", "r")}
           | {"japanese_mapped_clipped_q1_q3": [cal["result"]["dims"][d]["japanese_mapped_clipped_percentiles"][q]
                                                for q in ("25", "75")]}
           for d in ("valence", "arousal")}

    summaries = {v: {a: {q: summary([r[q] for r in rows[v][a]]) for q in QUANTITIES} for a in authors} for v in VERSIONS}
    pooled = {v: {q: {"min": min(r[q] for a in authors for r in rows[v][a]),
                      "max": max(r[q] for a in authors for r in rows[v][a])} for q in QUANTITIES} for v in VERSIONS}

    tmp = Path(tempfile.mkdtemp(prefix=".paper_generation_record-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {key: cm.sha256(p) for key, p in path.items()},
                        "library_sha256": cm.library_sha256(), **cm.versions("numpy"),
                        **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"C1_constants": {"authors": len(authors), "words_per_author": [len(seq[a]) for a in authors]},
                          "C2_G0_equals_words_json": c2,
                          "C3_values_in_range_no_clip": c3,
                          "C4_jacobian_f_from_A_only_s_from_V_only_equals_finite_differences": True,
                          "C5_fit_copied": True},
               "jacobian": {"rows": ["f", "s", "gamma"], "columns": ["dV", "dA"], "matrix": J.tolist()},
               "map_B_fit": fit,
               "summaries": summaries,
               "pooled": pooled,
               "words": rows}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        with (tmp / RECORD).open("x") as fh:
            fh.write(text)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(res["checks"]))


if __name__ == "__main__":
    main()
