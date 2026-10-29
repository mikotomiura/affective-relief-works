"""The share of the range of f and s used by the 104 sentences of work 1, next to comparisons of the same unit: the
matched terms (each occurrence and each distinct term) next to the whole NRC lexicon, and the sentences next to
sentences drawn at random from the lexicon with the same numbers of matched terms.

Usage:  uv run python works/paper_work1_same_unit.py   (refuses to overwrite)
Conditions: works/conditions/paper_work1_same_unit.json (committed before this script was written)
Input:  results/work1/work1.json, results/paper_used_range/used_range.json and the NRC lexicon of work 1 (no drawing)
Output: results/paper_work1_same_unit/
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
from atact import mapping  # noqa: E402
from atact.lexicon import load_lexicon  # noqa: E402

import _common as cm  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_work1_same_unit.json"
OUT_DIR = ROOT / "results" / "paper_work1_same_unit"
CODE = ["works/paper_work1_same_unit.py", "works/_common.py"]
QUANTITIES = ("f", "s")
SHARES = ("share_quartiles_percent", "share_min_max_percent")
RECORD = "same_unit.json"
CONSTANTS = {"sentences": 104, "matched_tokens": 1227, "distinct_terms": 583, "lexicon_terms": 54800,
             "replicates": 1000, "seed": 20261029}


def spread(xs: np.ndarray, width: float) -> dict:
    """Minimum, quartiles (linear), maximum and the two shares of the width of the range of the rule, as
    works/paper_used_range.py."""
    q1, q3 = (float(x) for x in np.quantile(xs, [0.25, 0.75], method="linear"))
    lo, hi = float(xs.min()), float(xs.max())
    return {"n": int(xs.size), "min": lo, "q1": q1, "q3": q3, "max": hi,
            "share_min_max_percent": 100 * (hi - lo) / width, "share_quartiles_percent": 100 * (q3 - q1) / width}


W4_PAIRS = [0]  # the pairs of V and A that passed shapes


def shapes(V, A) -> dict[str, np.ndarray]:
    """f and s of atact.mapping.params, after checking that params clips nothing (W4)."""
    V, A = np.asarray(V, dtype=float), np.asarray(A, dtype=float)
    if V.shape != A.shape or not V.size:
        raise SystemExit("W4: V and A are empty or of different sizes")
    if not (np.all((V >= -1.0) & (V <= 1.0)) and np.all((A >= -1.0) & (A <= 1.0))):
        raise SystemExit("W4: a V or A outside [-1, 1]")
    f, s, _ = mapping.params(V, A)
    W4_PAIRS[0] += int(V.size)
    return {"f": np.asarray(f, dtype=float), "s": np.asarray(s, dtype=float)}


def draw(seed: int, R: int, Z: list[int], LV: list[float], LA: list[float], width: dict) -> dict[str, dict[str, list]]:
    """R replicates of the 104 sentences drawn from the lexicon: per replicate one call of integers for all the draws,
    split by sentence in their order, V and A the mean of the drawn terms summed in order as work 1 sums them (W2).
    Returns the shares of every replicate."""
    rng = np.random.default_rng(seed)
    bounds = list(zip(np.concatenate([[0], np.cumsum(Z)[:-1]]).tolist(), np.cumsum(Z).tolist()))
    out = {q: {k: [] for k in SHARES} for q in QUANTITIES}
    for _ in range(R):
        idx = rng.integers(0, len(LV), size=sum(Z)).tolist()
        V = [sum(LV[i] for i in idx[a:b]) / (b - a) for a, b in bounds]
        A = [sum(LA[i] for i in idx[a:b]) / (b - a) for a, b in bounds]
        sh = shapes(V, A)
        for q in QUANTITIES:
            sp = spread(sh[q], width[q])
            for k in SHARES:
                out[q][k].append(sp[k])
    return out


def summary(xs: list[float], observed: float) -> dict:
    a = np.asarray(xs, dtype=float)
    lo, hi = (float(x) for x in np.quantile(a, [0.025, 0.975], method="linear"))
    return {"mean": float(a.mean()), "median": float(np.median(a)), "q0.025": lo, "q0.975": hi,
            "replicates_at_most_the_sentences": int((a <= observed).sum())}


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    k = CONSTANTS
    if c["constants"] != k or c["random"]["replicates"] != k["replicates"] or c["random"]["seed"] != k["seed"]:
        raise SystemExit("the constants of the conditions differ from those of the script")
    if tuple(c["statistics"]["quantities"]) != QUANTITIES or not c["outputs"]["record"].startswith(RECORD + ":"):
        raise SystemExit("the quantities or the record of the conditions differ from those of the script")
    inp = c["inputs"]
    path = {key: ROOT / v["path"] for key, v in inp.items() if isinstance(v, dict)}
    cm.same_as_head([COND, *path.values(), *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "results/work1", "results/paper_used_range", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    c1 = json.loads(path["work1_conditions"].read_bytes())
    w1 = json.loads(path["work1"].read_bytes())
    ur = json.loads(path["used_range"].read_bytes())
    for name, rec, cond in (("work1.json", w1, "work1_conditions"), ("used_range.json", ur, "used_range_conditions")):
        if rec["meta"]["conditions_sha256"] != cm.sha256(path[cond]) or rec["meta"].get("git_dirty") is not False:
            raise SystemExit(f"{name} was not made on its conditions on a clean tree")
    if ur["meta"]["inputs_sha256"]["work1"] != cm.sha256(path["work1"]):
        raise SystemExit("used_range.json was not made on this work1.json")
    cm.check_generator(c1["generator"])  # W4
    for name, rec in (("work1.json", w1), ("used_range.json", ur)):
        if rec["meta"]["atact_commit"] != cm.installed_commit():
            raise SystemExit(f"W4: {name} was made with another generator")
    nrc = load_lexicon(ROOT / c1["lexicon"]["file"])
    if nrc.sha256 != c1["lexicon"]["sha256"] or w1["meta"]["lexicon_sha256"] != nrc.sha256 or ur["meta"]["nrc_sha256"] != nrc.sha256:
        raise SystemExit("the NRC lexicon differs from that of work 1 or of used_range.json")
    rule = {"f": (mapping.F0, mapping.F0 + mapping.F1), "s": (mapping.S0, mapping.S0 + mapping.S1)}  # W4
    if not all(np.allclose(rule[q], want, rtol=0.0, atol=1e-12) for q, want in (("f", (3.0, 7.0)), ("s", (0.06, 0.30)))):
        raise SystemExit(f"W4: the range of the rule {rule} is not the one of the conditions")
    width = {q: hi - lo for q, (lo, hi) in rule.items()}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # W1: the constants ---------------------------------------------------
    sents = w1["sentences"]
    if len(sents) != k["sentences"] or w1["unmatched"] != 0 or any(p["unmatched"] for p in sents):
        raise SystemExit("W1: work 1 has not 104 sentences all matched")
    if any(p["Z"] != len(p["matched"]) or not p["matched"] for p in sents):
        raise SystemExit("W1: a Z differs from the number of matched terms")
    tokens = [t for p in sents for t in p["matched"]]
    types = sorted(set(tokens))
    terms = sorted(nrc.entries)
    if len(tokens) != k["matched_tokens"] or len(types) != k["distinct_terms"] or len(terms) != k["lexicon_terms"]:
        raise SystemExit(f"W1: {len(tokens)} tokens, {len(types)} terms, {len(terms)} terms of the lexicon")

    # W2: the sentences are the means of their matched terms ---------------
    w2 = 0
    for i, p in enumerate(sents):
        if any(t not in nrc.entries for t in p["matched"]):
            raise SystemExit(f"W2: sentence {i}: a matched term is not an entry of the lexicon")
        z = len(p["matched"])
        V = sum(nrc.entries[t][0] for t in p["matched"]) / z
        A = sum(nrc.entries[t][1] for t in p["matched"]) / z
        if (p["V"], p["A"]) != (V, A):
            raise SystemExit(f"W2: sentence {i}: V and A differ from the mean of its matched terms")
        if tuple(float(x) for x in mapping.params(V, A)[:2]) != (p["f"], p["s"]):
            raise SystemExit(f"W2: sentence {i}: f or s differ from params")
        w2 += 1

    # the sets ----------------------------------------------------------------
    def of(ts: list[str]) -> dict[str, np.ndarray]:
        return shapes([nrc.entries[t][0] for t in ts], [nrc.entries[t][1] for t in ts])

    def table(sh: dict[str, np.ndarray]) -> dict:
        return {q: spread(sh[q], width[q]) for q in QUANTITIES}

    sentences = table(shapes([p["V"] for p in sents], [p["A"] for p in sents]))
    lexicon = table(of(terms))
    # W3: the same as used_range.json ------------------------------------------
    if sentences != ur["sets"]["work1_sentences"] or lexicon != ur["sets"]["work1_lexicon"]:
        raise SystemExit("W3: the sentences or the lexicon differ from used_range.json")
    w3 = sum(len(x[q]) for x in (sentences, lexicon) for q in QUANTITIES)
    sets = {"sentences": sentences, "terms_tokens": table(of(tokens)), "terms_types": table(of(types)),
            "lexicon": lexicon}

    # the random sentences and W5 ---------------------------------------------
    Z = [p["Z"] for p in sents]
    LV = [nrc.entries[t][0] for t in terms]
    LA = [nrc.entries[t][1] for t in terms]
    reps = draw(k["seed"], k["replicates"], Z, LV, LA, width)
    again = draw(k["seed"], k["replicates"], Z, LV, LA, width)
    if reps != again or any(len(reps[q][s]) != k["replicates"] for q in QUANTITIES for s in SHARES):
        raise SystemExit("W5: the draws are not reproduced or not of the number of replicates")
    if any(len(set(reps[q][s])) < 2 for q in QUANTITIES for s in SHARES):
        raise SystemExit("W5: every replicate gives the same share (the generator is not shared across replicates)")
    random_sentences = {q: {s: summary(reps[q][s], sentences[q][s]) for s in SHARES} for q in QUANTITIES}

    tmp = Path(tempfile.mkdtemp(prefix=".paper_work1_same_unit-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {key: cm.sha256(p) for key, p in path.items()},
                        "nrc_sha256": nrc.sha256, "atact_commit": cm.installed_commit(),
                        "library_sha256": cm.library_sha256(), **cm.versions("numpy"),
                        **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"W1_constants": CONSTANTS, "W2_sentences_checked": w2,
                          "W3_fields_equal_used_range": w3, "W4_pairs_within_range": W4_PAIRS[0],
                          "W5_draws_reproduced": k["replicates"], "nrc_skipped": list(nrc.skipped)},
               "rule_range": {q: list(v) for q, v in rule.items()},
               "sets": sets,
               "random_sentences": {"summary": random_sentences, "replicates": reps}}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        with (tmp / RECORD).open("x", encoding="utf-8") as fh:
            fh.write(text)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(res["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
