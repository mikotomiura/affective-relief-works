"""The share of the range of f and s used by the 104 sentences of work 1, beside the story's own matched tokens shuffled
among the sentences (each sentence keeping its number of matched terms) and the uniform draws of
works/paper_work1_same_unit.py as recorded.

Usage:  uv run python works/paper_work1_shuffled.py   (refuses to overwrite)
Conditions: works/conditions/paper_work1_shuffled.json (committed before this script was written)
Input:  results/work1/work1.json, results/paper_work1_same_unit/same_unit.json and the NRC lexicon of work 1 (no drawing)
Output: results/paper_work1_shuffled/
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from atact import mapping  # noqa: E402
from atact.lexicon import load_lexicon  # noqa: E402

import _common as cm  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_work1_shuffled.json"
SAME_SCRIPT = ROOT / "works" / "paper_work1_same_unit.py"
OUT_DIR = ROOT / "results" / "paper_work1_shuffled"
CODE = ["works/paper_work1_shuffled.py", "works/paper_work1_same_unit.py", "works/_common.py"]
QUANTITIES = ("f", "s")
SHARES = ("share_quartiles_percent", "share_min_max_percent")
RECORD = "shuffled.json"
CONSTANTS = {"sentences": 104, "matched_tokens": 1227, "replicates": 1000, "seed": 20261030}


def shuffle(seed: int, R: int, tokens: list[str], Z: list[int], entries: dict, width: dict, su) -> dict:
    """R replicates: one permutation of the tokens per replicate, split by sentence in their order with each sentence
    taking its own Z, V and A the mean of its tokens summed in order (as work 1). Returns the shares of every
    replicate and checks X3 on each."""
    if sum(Z) != len(tokens):  # X3
        raise SystemExit("X3: the numbers of matched terms do not add up to the tokens")
    rng = np.random.default_rng(seed)
    bounds = list(zip(np.concatenate([[0], np.cumsum(Z)[:-1]]).tolist(), np.cumsum(Z).tolist()))
    want = Counter(tokens)
    out = {q: {k: [] for k in SHARES} for q in QUANTITIES}
    out["x3"] = 0
    for _ in range(R):
        perm = rng.permutation(len(tokens)).tolist()
        toks = [tokens[i] for i in perm]
        if sorted(perm) != list(range(len(tokens))) or Counter(toks) != want or bounds[-1][1] != len(toks):  # X3
            raise SystemExit("X3: a replicate is not a permutation of the tokens with the Z of the sentences")
        out["x3"] += 1
        V = [sum(entries[t][0] for t in toks[a:b]) / (b - a) for a, b in bounds]
        A = [sum(entries[t][1] for t in toks[a:b]) / (b - a) for a, b in bounds]
        sh = su.shapes(V, A)  # X4
        for q in QUANTITIES:
            sp = su.spread(sh[q], width[q])
            for k in SHARES:
                out[q][k].append(sp[k])
    return out


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
    if path["same_unit_script"] != SAME_SCRIPT:
        raise SystemExit("the script of the conditions is not the one this script imports")
    cm.same_as_head([COND, *path.values(), *(ROOT / p for p in CODE)])  # before anything is imported from it
    state = cm.git_state(["works", "src", "results/work1", "results/paper_work1_same_unit", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    import paper_work1_same_unit as su

    if Path(su.__file__).resolve() != SAME_SCRIPT.resolve():
        raise SystemExit(f"the module imported is {su.__file__}, not {SAME_SCRIPT}")
    c1 = json.loads(path["work1_conditions"].read_bytes())
    w1 = json.loads(path["work1"].read_bytes())
    sa = json.loads(path["same_unit"].read_bytes())
    for name, rec, cond in (("work1.json", w1, "work1_conditions"), ("same_unit.json", sa, "same_unit_conditions")):
        if rec["meta"]["conditions_sha256"] != cm.sha256(path[cond]) or rec["meta"].get("git_dirty") is not False:
            raise SystemExit(f"{name} was not made on its conditions on a clean tree")
    if (sa["meta"]["script_sha256"] != cm.sha256(SAME_SCRIPT)
            or sa["meta"]["inputs_sha256"]["work1"] != cm.sha256(path["work1"])):
        raise SystemExit("same_unit.json was not made by that script from this work1.json")
    cm.check_generator(c1["generator"])  # X4
    for name, rec in (("work1.json", w1), ("same_unit.json", sa)):
        if rec["meta"]["atact_commit"] != cm.installed_commit():
            raise SystemExit(f"X4: {name} was made with another generator")
    nrc = load_lexicon(ROOT / c1["lexicon"]["file"])
    if nrc.sha256 != c1["lexicon"]["sha256"] or sa["meta"]["nrc_sha256"] != nrc.sha256:
        raise SystemExit("the NRC lexicon differs from that of work 1 or of same_unit.json")
    rule = {"f": (mapping.F0, mapping.F0 + mapping.F1), "s": (mapping.S0, mapping.S0 + mapping.S1)}
    if not all(np.allclose(rule[q], want, rtol=0.0, atol=1e-12) for q, want in (("f", (3.0, 7.0)), ("s", (0.06, 0.30)))):
        raise SystemExit(f"X4: the range of the rule {rule} is not the one of the conditions")
    width = {q: hi - lo for q, (lo, hi) in rule.items()}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # X1 ------------------------------------------------------------------------
    sents = w1["sentences"]
    if len(sents) != k["sentences"] or any(p["Z"] != len(p["matched"]) or not p["matched"] for p in sents):
        raise SystemExit("X1: not 104 sentences with Z their number of matched terms")
    tokens = [t for p in sents for t in p["matched"]]
    if len(tokens) != k["matched_tokens"] or any(t not in nrc.entries for t in tokens):
        raise SystemExit(f"X1: {len(tokens)} tokens, or a token not in the lexicon")
    Z = [p["Z"] for p in sents]

    # X2 ------------------------------------------------------------------------
    for i, p in enumerate(sents):
        z = len(p["matched"])
        V = sum(nrc.entries[t][0] for t in p["matched"]) / z
        A = sum(nrc.entries[t][1] for t in p["matched"]) / z
        if (p["V"], p["A"]) != (V, A) or tuple(float(x) for x in mapping.params(V, A)[:2]) != (p["f"], p["s"]):
            raise SystemExit(f"X2: sentence {i} differs from the mean of its matched terms")
    sh = su.shapes([p["V"] for p in sents], [p["A"] for p in sents])
    sentences = {q: su.spread(sh[q], width[q]) for q in QUANTITIES}
    if sentences != sa["sets"]["sentences"]:
        raise SystemExit("X2: the shares of the sentences differ from same_unit.json")

    # the shuffles and X5 ---------------------------------------------------------
    pairs0 = su.W4_PAIRS[0]
    reps = shuffle(k["seed"], k["replicates"], tokens, Z, nrc.entries, width, su)
    again = shuffle(k["seed"], k["replicates"], tokens, Z, nrc.entries, width, su)
    x4 = su.W4_PAIRS[0] - pairs0
    x3 = reps.pop("x3") + again.pop("x3")
    if x3 != 2 * k["replicates"] or x4 != 2 * k["replicates"] * k["sentences"]:
        raise SystemExit(f"X3/X4: {x3} replicates and {x4} pairs checked")
    if reps != again or any(len(reps[q][s]) != k["replicates"] for q in QUANTITIES for s in SHARES):
        raise SystemExit("X5: the shuffles are not reproduced or not of the number of replicates")
    if any(len(set(reps[q][s])) < 2 for q in QUANTITIES for s in SHARES):
        raise SystemExit("X5: every replicate gives the same share")
    summary = {q: {s: su.summary(reps[q][s], sentences[q][s]) for s in SHARES} for q in QUANTITIES}

    tmp = Path(tempfile.mkdtemp(prefix=".paper_work1_shuffled-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                        "inputs_sha256": {key: cm.sha256(p) for key, p in path.items()},
                        "nrc_sha256": nrc.sha256, "atact_commit": cm.installed_commit(),
                        "library_sha256": cm.library_sha256(), **cm.versions("numpy"),
                        **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"X1_constants": CONSTANTS, "X2_sentences_checked": len(sents), "X3_replicates_checked": x3,
                          "X4_pairs_within_range_in_the_shuffles": x4,
                          "X5_replicates_compared": len(reps["f"]["share_quartiles_percent"]),
                          "nrc_skipped": list(nrc.skipped)},
               "rule_range": {q: list(v) for q, v in rule.items()},
               "sentences": sentences,
               "shuffled_sentences": {"summary": summary, "replicates": reps},
               "uniform_draws_as_recorded": sa["random_sentences"]["summary"]}
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
