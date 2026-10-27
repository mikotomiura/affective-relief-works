"""Check that the generator pinned in pyproject.toml gives the stored values of the text example of the paper on the rule.

Usage:  uv run python checks/reproduce_text_example.py            (compute, compare, write the result; refuses to overwrite)
Conditions: checks/conditions/reproduce_text_example.json (committed before this script was run)
Input:  checks/reference/text_path_example.json (results/text_path_example.json of the public repository) and the
        NRC VAD Lexicon v2.1 (external/, not in the repository)
Output: results/reproduce_text_example.json

The point and the tile of each sentence follow point() and tile() of experiments/text_path_example.py of the public
repository at the pinned commit; they are written out here because that script is not part of the installed package.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from atact.field import procedural_field_separable
from atact.lexicon import Lexicon, load_lexicon, match_terms, split_sentences, tokenize
from atact.mapping import params
from atact.transform import t_gamma, t_gamma_eps

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
COND = ROOT / "checks" / "conditions" / "reproduce_text_example.json"
OUT = ROOT / "results" / "reproduce_text_example.json"
OCTAVES = 4
EXPORTS = ("common", "minmax", "common_eps")
EXACT = ("matched", "Z", "unmatched")
LIB_FILES = ("field.py", "lexicon.py", "mapping.py", "transform.py")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def installed_commit() -> str:
    d = json.loads(metadata.distribution("atact").read_text("direct_url.json"))
    return d["vcs_info"]["commit_id"]


def library_sha256() -> dict[str, str]:
    """SHA-256 of the installed library files that the script uses (the commit in direct_url.json alone does not
    show that the installed files were not changed)."""
    import atact

    d = Path(atact.__file__).parent
    return {n: sha256(d / n) for n in LIB_FILES}


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    paths = ["checks", "pyproject.toml", "uv.lock", ".python-version"]
    commit = run("rev-parse", "HEAD")
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise SystemExit(f"unexpected commit from git: {commit!r}")
    return {"git_commit": commit, "git_dirty": bool(run("status", "--porcelain", "--", *paths))}


def check_inputs(c: dict) -> tuple[dict, Lexicon]:
    if installed_commit() != c["generator"]["commit"]:
        raise SystemExit(f"atact is installed from {installed_commit()}, not {c['generator']['commit']}")
    if library_sha256() != c["generator"]["library_sha256"]:
        raise SystemExit("the installed library files differ from those of the commit named in the conditions")
    got = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__}
    for k, v in got.items():
        if v != c["versions"][k]:
            raise SystemExit(f"{k} is {v}, not {c['versions'][k]} as in the conditions")
    ref_path = ROOT / c["reference"]["file"]
    if sha256(ref_path) != c["reference"]["sha256"]:
        raise SystemExit(f"{ref_path} differs from the reference named in the conditions")
    lex = load_lexicon(ROOT / c["lexicon"]["file"].split(" ")[0])
    # the hash of the bytes that load_lexicon parsed, not of a second read of the file
    if lex.sha256 != c["lexicon"]["sha256"]:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    return json.loads(ref_path.read_text()), lex


def point(sentence: str, lex: Lexicon, drop: frozenset[str] | None) -> dict:
    terms = match_terms(tokenize(sentence), lex)
    if drop is not None:
        terms = [t for t in terms if not all(w in drop for w in t.split(" "))]
    z = len(terms)
    V = sum(lex.entries[t][0] for t in terms) / z if z else 0.0
    A = sum(lex.entries[t][1] for t in terms) / z if z else 0.0
    f, s, g = (float(x) for x in params(V, A))
    return {"text": sentence, "matched": terms, "Z": z, "unmatched": z == 0, "V": V, "A": A, "f": f,
            "f_frac": float(f - np.floor(f)), "s": s, "gamma": g}


def max_grad_edge(X: np.ndarray) -> tuple[float, bool]:
    n = X.shape[0]
    gy, gx = np.gradient(X, 1.0 / n)
    m = np.hypot(gx, gy)
    j, i = np.unravel_index(int(np.argmax(m)), m.shape)
    return float(m[j, i]), bool(i in (0, n - 1) or j in (0, n - 1))


def minmax(X: np.ndarray) -> np.ndarray:
    return (X - X.min()) / (X.max() - X.min())


def tile(p: dict, n: int, seed: int, eps: float, d: dict) -> dict:
    D, L, s_max = d["D"], d["L"], d["s_max"]
    F = procedural_field_separable(p["f"], n, seed, octaves=OCTAVES, normalize="ampsum")
    T = t_gamma(F, p["gamma"])
    H, H_eps = p["s"] * T, p["s"] * t_gamma_eps(F, p["gamma"], eps)
    k = D / (2 * s_max)
    z = {"common": k * H, "minmax": D * minmax(H), "common_eps": k * H_eps}
    scale = {"common": k, "minmax": D / (H.max() - H.min()), "common_eps": k}
    src = {"common": H, "minmax": H, "common_eps": H_eps}
    G = max_grad_edge(F)[0]
    out = {"G": G,
           "angle_deg": {e: float(np.degrees(np.arctan(scale[e] * max_grad_edge(src[e])[0] / L))) for e in EXPORTS},
           "slope": {e: scale[e] * max_grad_edge(src[e])[0] / L for e in EXPORTS},
           "edge_argmax": {e: max_grad_edge(src[e])[1] for e in EXPORTS},
           "span_mm": {e: float(z[e].max() - z[e].min()) for e in EXPORTS},
           "shape_change_mm": float(np.abs(z["common_eps"] - z["common"]).max()),
           "pt1_diff": float(np.abs(z["minmax"] / D - (T - T.min()) / (T.max() - T.min())).max()),
           "pt2_common_expected": float(k * p["s"] * (T.max() - T.min()))}
    if p["gamma"] > 1:
        se = eps ** (1.0 / p["gamma"] - 1.0) * (1.0 + eps**2) ** ((1.0 - 1.0 / p["gamma"]) / 2.0)
        out["pt3_slope_bound"] = k * p["s"] * se * G / L
        out["pt3_shape_bound_mm"] = k * p["s"] * eps ** (1.0 / p["gamma"])
    return out


def compare(ref, got, path: str, diffs: list, mismatches: list) -> None:
    """Walk the stored value; every stored number must be matched by the recomputed one.

    Dicts are compared key by key and lists element by element. Strings, booleans and the items of EXACT must be
    equal and of the same type (so 3 and 3.0, or 1 and True, differ). Other numbers must be finite on both sides and
    not booleans; their absolute differences are collected in diffs and judged one by one by the caller."""
    if isinstance(ref, dict) or isinstance(got, dict):
        if not (isinstance(ref, dict) and isinstance(got, dict)) or set(ref) != set(got):
            mismatches.append(f"{path}: keys {sorted(ref) if isinstance(ref, dict) else ref!r} vs "
                              f"{sorted(got) if isinstance(got, dict) else got!r}")
            return
        for k in ref:
            compare(ref[k], got[k], f"{path}.{k}", diffs, mismatches)
    elif isinstance(ref, list) or isinstance(got, list):
        if not (isinstance(ref, list) and isinstance(got, list)) or len(ref) != len(got):
            mismatches.append(f"{path}: {ref!r} vs {got!r}")
            return
        for i, (r, g) in enumerate(zip(ref, got)):
            compare(r, g, f"{path}[{i}]", diffs, mismatches)
    elif isinstance(ref, (str, bool)) or isinstance(got, (str, bool)) or path.rsplit(".", 1)[-1] in EXACT:
        if type(ref) is not type(got) or ref != got:
            mismatches.append(f"{path}: {ref!r} vs {got!r}")
    elif isinstance(ref, (int, float)) and isinstance(got, (int, float)) and math.isfinite(ref) and math.isfinite(got):
        diffs.append((abs(float(ref) - float(got)), path))
    else:
        mismatches.append(f"{path}: not a pair of finite numbers: {ref!r} vs {got!r}")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    c = json.loads(COND.read_text())
    ref, lex = check_inputs(c)
    state = git_state()
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    a = ref["analysis"]
    drops = {"main": None, "function_words_removed": frozenset(ENGLISH_STOP_WORDS)}
    diffs: list = []
    mismatches: list = []
    n_rows = 0
    for name, t in a["texts"].items():
        for v, rows in t["variants"].items():
            got_rows = [point(s, lex, drops[v]) for s in split_sentences(t["content"])]
            if len(got_rows) != len(rows):
                mismatches.append(f"{name}.{v}: {len(got_rows)} sentences, stored {len(rows)}")
                continue
            for i, (r, p) in enumerate(zip(rows, got_rows)):
                p["grids"] = {str(n): tile(p, n, a["seed"], a["eps"], a["dimensions"]) for n in a["grids_N"]}
                compare(r, p, f"{name}.{v}[{i}]", diffs, mismatches)
                n_rows += 1
    worst = max(diffs, key=lambda d_: d_[0]) if diffs else (0.0, "")
    over = [p_ for d_, p_ in diffs if not d_ <= c["tolerance"]]
    passed = (not mismatches and not over and n_rows == c["expected"]["sentence_variant_rows"]
              and len(diffs) == c["expected"]["numbers_compared"])

    out = {"meta": {"conditions_sha256": sha256(COND), "script_sha256": sha256(SCRIPT),
                    "uv_lock_sha256": sha256(ROOT / "uv.lock"), "pyproject_sha256": sha256(ROOT / "pyproject.toml"),
                    "atact_commit": installed_commit(), "atact_library_sha256": library_sha256(),
                    "reference_sha256": sha256(ROOT / c["reference"]["file"]), "lexicon_sha256": lex.sha256,
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__, **state, "platform": f"{platform.system()} {platform.machine()}",
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "result": {"sentence_variant_rows": n_rows, "numbers_compared": len(diffs),
                      "max_abs_diff": worst[0], "max_abs_diff_at": worst[1],
                      "n_exact_diff_zero": sum(1 for d_, _ in diffs if d_ == 0.0),
                      "over_tolerance": over, "mismatches": mismatches, "tolerance": c["tolerance"],
                      "passed": passed}}
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("x") as f:  # exclusive creation: never overwrites a result written in the meantime
        f.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["result"], indent=1))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
