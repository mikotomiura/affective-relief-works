"""The share of the range of f and s used on the reference sets of the two versions of work 2 (the whole lexicon under
map B, the tokens of the 23 works with each author weighing 1 under the percentiles and under map B) and by the 20 words
of each trial author, minimum to maximum and quartiles.

Usage:  uv run python works/paper_reference_range.py   (refuses to overwrite)
Conditions: works/conditions/paper_reference_range.json (committed before this script was written)
Input:  the 23 works of work 2 (as works/sketch_sp_color.py), the lexicon of work 2 and the committed records named in
        the conditions (no drawing)
Output: results/paper_reference_range/
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from atact import mapping  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_reference_range.json"
OUT_DIR = ROOT / "results" / "paper_reference_range"
CODE = ["works/paper_reference_range.py", "works/_common.py", "src/pipeline.py", "src/japanese_text.py"]
VERSIONS, QUANTITIES = ("G0", "P"), ("f", "s")
RECORD = "reference_range.json"
CONSTANTS = {"authors_reference": 6, "trial_authors": 2, "words_per_author": 20, "works": 23,
             "lexicon_entries": 2003306, "reference_words": 18301}


def weighted_percentile(values: np.ndarray, weights: np.ndarray, x: float) -> float:
    """Copied from works/sketch_sp_color.py: weight of the values below x plus half the weight of those equal to x, over
    the total weight."""
    total = weights.sum()
    if not total > 0:
        raise ValueError("no weight")
    return float((weights[values < x].sum() + 0.5 * weights[values == x].sum()) / total)


def reference_weights(counts: dict[str, Counter]) -> dict[str, dict[str, float]]:
    """Copied from works/sketch_sp_color.py: per word, three weights (per author equal, tokens, types)."""
    out: dict[str, dict[str, float]] = {}
    for a, cnt in counts.items():
        T = sum(cnt.values())
        if T == 0:
            raise ValueError(f"{a}: no token")
        for w, n in cnt.items():
            d = out.setdefault(w, {"author_equal": 0.0, "tokens": 0.0, "types": 1.0})
            d["author_equal"] += n / T
            d["tokens"] += n
    return out


def nearest_index(sorted_xs: np.ndarray, q: int) -> float:
    """The percentile of calibrate_memolon.json: the value at index round(q/100 (n - 1)) of the sorted values."""
    return float(sorted_xs[round(q / 100 * (len(sorted_xs) - 1))])


def spread(xs: np.ndarray, width: float, weights: np.ndarray | None = None) -> dict:
    """Minimum, quartiles, maximum and the two shares of the width of the range of the rule."""
    if weights is None:
        q1, q3 = (float(x) for x in np.quantile(xs, [0.25, 0.75], method="linear"))
        extra = {}
    else:
        if xs.shape != weights.shape or not (weights > 0).all():
            raise SystemExit("weights of another shape or not all positive")
        q1, q3 = (float(x) for x in np.quantile(xs, [0.25, 0.75], method="inverted_cdf", weights=weights))
        extra = {"total_weight": float(weights.sum())}
    lo, hi = float(xs.min()), float(xs.max())
    return {"n": int(xs.size), **extra, "min": lo, "q1": q1, "q3": q3, "max": hi,
            "share_min_max_percent": 100 * (hi - lo) / width, "share_quartiles_percent": 100 * (q3 - q1) / width}


def shapes(V: np.ndarray, A: np.ndarray) -> dict[str, np.ndarray]:
    """f and s of atact.mapping.params, after checking that params clips nothing (R5)."""
    if not (np.all((V >= -1.0) & (V <= 1.0)) and np.all((A >= -1.0) & (A <= 1.0))):
        raise SystemExit("R5: a V or A outside [-1, 1]")
    f, s, _ = mapping.params(V, A)
    return {"f": np.asarray(f, dtype=float), "s": np.asarray(s, dtype=float)}


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    if c["constants"] != CONSTANTS:
        raise SystemExit("the constants of the conditions differ from those of the script")
    if tuple(c["statistics"]["quantities"]) != QUANTITIES or not c["outputs"]["record"].startswith(RECORD + ":"):
        raise SystemExit("the quantities or the record of the conditions differ from those of the script")
    inp = c["inputs"]
    path = {k: ROOT / v["path"] for k, v in inp.items() if isinstance(v, dict)}
    cm.check_file(path["values"], inp["values"]["sha256"], "values")
    w2 = json.loads(path["work2_conditions"].read_bytes())
    cm.same_as_head([COND, *path.values(), ROOT / w2["map"]["file"], *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "results/work2", "results/sketch_sp_color", "results/paper_generation_record",
                          "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    wc = json.loads(path["words_conditions"].read_bytes())
    rc = json.loads(path["reference_conditions"].read_bytes())
    gc = json.loads(path["generation_conditions"].read_bytes())
    gen = json.loads(path["generation_record"].read_bytes())
    cal_raw = cm.check_file(path["calibration"], w2["map"]["sha256"], "the calibration (the map of work 2)")
    cal = json.loads(cal_raw)
    if cal["meta"].get("git_dirty") is not False:
        raise SystemExit("calibrate_memolon.json was not computed on a clean tree")
    if (gen["meta"]["conditions_sha256"] != cm.sha256(path["generation_conditions"]) or gen["meta"].get("git_dirty") is not False
            or gen["meta"]["inputs_sha256"]["values"] != inp["values"]["sha256"]
            or gc["inputs"]["values"]["sha256"] != inp["values"]["sha256"]):
        raise SystemExit("the record of generation was not made on its conditions and the values named, on a clean tree")
    if wc["work2_conditions_sha256"] != cm.sha256(path["work2_conditions"]):
        raise SystemExit("work2.json is not the one the words were selected on")
    if rc["inputs"]["work2_conditions"] != "works/conditions/work2.json" or rc["reference"]["select"] != "all":
        raise SystemExit("the reference of sketch_sp_color is not the one of these conditions")
    cm.check_generator(wc["generator"])  # R6
    if gen["meta"]["library_sha256"] != cm.library_sha256() or gen["meta"]["atact_commit"] != cm.installed_commit():
        raise SystemExit("R6: the record of generation was made with another generator")
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of work 2")
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if lex.sha256 != w2["lexicon"]["sha256"]:
        raise SystemExit("the lexicon differs from that of work 2")
    vmap = calibration_map(cal_raw)
    rule = {"f": (mapping.F0, mapping.F0 + mapping.F1), "s": (mapping.S0, mapping.S0 + mapping.S1)}  # R6
    if not all(np.allclose(rule[q], want, rtol=0.0, atol=1e-12) for q, want in (("f", (3.0, 7.0)), ("s", (0.06, 0.30)))):
        raise SystemExit(f"R6: the range of the rule {rule} is not the one of the conditions")
    width = {q: hi - lo for q, (lo, hi) in rule.items()}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # R1: the constants ---------------------------------------------------
    k = CONSTANTS
    authors = list(gen["words"]["G0"])
    if (len(w2["works"]) != k["works"] or len({w["author"] for w in w2["works"]}) != k["authors_reference"]
            or len(lex.entries) != k["lexicon_entries"] or cal["result"]["japanese_entries"] != k["lexicon_entries"]):
        raise SystemExit("R1: the works, authors or entries of the lexicon are not the constants")
    if (authors != wc["trial_authors"] or len(authors) != k["trial_authors"]
            or any(list(gen["words"][v]) != authors or len(gen["words"][v][a]) != k["words_per_author"]
                   for v in VERSIONS for a in authors)):
        raise SystemExit("R1: the trial authors or their words in the record of generation are not the constants")

    # the whole lexicon under map B, R2 and R5 ----------------------------
    raw_v = np.fromiter((v for v, _ in lex.entries.values()), dtype=float, count=len(lex.entries))
    raw_a = np.fromiter((a for _, a in lex.entries.values()), dtype=float, count=len(lex.entries))
    mapped = {"valence": vmap.b_v * raw_v + vmap.a_v, "arousal": vmap.b_a * raw_a + vmap.a_a}
    if not all(np.isfinite(x).all() for x in mapped.values()):
        raise SystemExit("a mapped value of the lexicon is not finite")
    clipped = {d: np.clip(x, -1.0, 1.0) for d, x in mapped.items()}
    clip_count = {d: int((mapped[d] != clipped[d]).sum()) for d in mapped}
    r2 = 0
    for d, x in clipped.items():
        xs = np.sort(x)
        rec = cal["result"]["dims"][d]["japanese_mapped_clipped_percentiles"]
        for q in (0, 25, 75, 100):
            if nearest_index(xs, q) != rec[str(q)]:
                raise SystemExit(f"R2: the percentile {q} of the mapped {d} is not that of calibrate_memolon.json")
            r2 += 1
        if (float(xs[0]), float(xs[-1])) != (rec["0"], rec["100"]):
            raise SystemExit(f"R2: the minimum or maximum of the mapped {d} is not that of calibrate_memolon.json")
    lexicon_shapes = shapes(clipped["valence"], clipped["arousal"])

    # the reference of the 23 works, R3 -----------------------------------
    done = {w["work_id"]: w for ws in json.loads(path["work2_result"].read_text())["authors"].values() for w in ws}
    counts: dict[str, Counter] = {}
    for w in w2["works"]:
        cm.check_file(ROOT / w["zip"], w["zip_sha256"], f"the zip of {w['work_id']}")
        name, data = read_aozora_zip(ROOT / w["zip"])
        if name != w["txt_name"]:
            raise SystemExit(f"{w['work_id']}: the zip holds {name}")
        text = data.decode(w["encoding"])
        body, _, _, kept, _ = work2_body(text)
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != w["body_sha256"] or len(kept) != w["sentences"]:
            raise SystemExit(f"{w['work_id']}: the body differs from work 2")
        t = work2_text(text, lex, tagger, vmap)
        ref = done[w["work_id"]]
        if len(t.points) != ref["sentences"] or sum(p.Z for p in t.points) != ref["matched_tokens"]:
            raise SystemExit(f"R3: {w['work_id']}: the text step gives another result than results/work2")
        everything = japanese_text.text_path(t.sentences, lex, tagger, vmap, select=rc["reference"]["select"], order="word")
        counts.setdefault(w["author"], Counter()).update(x for p in everything for x in p.matched)
    values_rec = json.loads(path["values"].read_bytes())
    if (len(counts) != k["authors_reference"]
            or {a: sum(n.values()) for a, n in counts.items()} != values_rec["reference_tokens"]):
        raise SystemExit("R1: the tokens per author differ from those of values.json")
    wts = reference_weights(counts)
    ref_words = sorted(wts)
    if len(ref_words) != k["reference_words"] or values_rec["reference_words"] != k["reference_words"]:
        raise SystemExit(f"R1: {len(ref_words)} reference words, not {k['reference_words']}")
    rv = np.array([lex.entries[w][0] for w in ref_words])
    ra = np.array([lex.entries[w][1] for w in ref_words])
    weight = np.array([wts[w]["author_equal"] for w in ref_words])
    r3 = 0
    for key, v in values_rec["values"].items():
        e = v["entry"]
        got = (weighted_percentile(rv, weight, lex.entries[e][0]), weighted_percentile(ra, weight, lex.entries[e][1]))
        if got != (v["p_valence_author_equal"], v["p_arousal_author_equal"]):
            raise SystemExit(f"R3: the percentiles of {key} recomputed differ from values.json")
        r3 += 1
    if abs(weight.sum() - k["authors_reference"]) > 1e-9:  # each author weighs 1
        raise SystemExit(f"the weights of the reference sum to {weight.sum()}, not {k['authors_reference']}")
    # the weighted percentile of every reference word among the reference, by the copied function itself
    p_v = np.array([weighted_percentile(rv, weight, x) for x in rv])
    p_a = np.array([weighted_percentile(ra, weight, x) for x in ra])
    tokens_P = shapes(2 * p_v - 1, 2 * p_a - 1)
    tokens_B = shapes(*(np.array([vmap(*lex.entries[w])[i] for w in ref_words]) for i in (0, 1)))
    # per author the weight n / T; the quantiles are taken with the integer n, which gives the same quantiles (they do
    # not depend on a common factor) without rounding a cumulative share that is exactly 1/4 or 3/4
    author_tokens = {a: np.array([cnt.get(w, 0) for w in ref_words], dtype=float) for a, cnt in counts.items()}
    if not np.allclose(sum(n_a / n_a.sum() for n_a in author_tokens.values()), weight, rtol=0.0, atol=1e-12):
        raise SystemExit("the weights n / T per author do not add up to the weights of the reference")

    # the words, R4 --------------------------------------------------------
    r4 = 0
    words_shapes: dict = {v: {} for v in VERSIONS}
    for version in VERSIONS:
        for a in authors:
            rows = gen["words"][version][a]
            for r in rows:
                e = values_rec["values"][r["word"]]["entry"]
                if e != r["word"]:
                    raise SystemExit(f"R4: the entry of {r['word']} is {e}")
                want = (vmap(*lex.entries[e]) if version == "G0"
                        else (2 * values_rec["values"][e]["p_valence_author_equal"] - 1,
                              2 * values_rec["values"][e]["p_arousal_author_equal"] - 1))
                if (r["V"], r["A"]) != tuple(float(x) for x in want):
                    raise SystemExit(f"R4: {version} {r['word']}: V and A differ")
                f, s, _ = (float(x) for x in mapping.params(r["V"], r["A"]))
                if (f, s) != (r["f"], r["s"]):
                    raise SystemExit(f"R4: {version} {r['word']}: f or s differ from params")
                r4 += 1
            computed = shapes(np.array([r["V"] for r in rows]), np.array([r["A"] for r in rows]))  # R5
            words_shapes[version][a] = {q: np.array([r[q] for r in rows], dtype=float) for q in QUANTITIES}  # as recorded
            if any(not np.array_equal(words_shapes[version][a][q], computed[q]) for q in QUANTITIES):
                raise SystemExit(f"R4: {version} {a}: the f or s taken differ from params of the V and A")
        words_shapes[version]["both"] = {q: np.concatenate([words_shapes[version][a][q] for a in authors])
                                         for q in QUANTITIES}
    if r4 != len(VERSIONS) * k["trial_authors"] * k["words_per_author"]:
        raise SystemExit(f"R4: {r4} words checked")

    def table(sh: dict[str, np.ndarray], wt: np.ndarray | None = None) -> dict:
        if wt is None:
            return {q: spread(sh[q], width[q]) for q in QUANTITIES}
        keep = wt > 0
        return {q: spread(sh[q][keep], width[q], wt[keep]) for q in QUANTITIES}

    def per_author(sh: dict[str, np.ndarray], n_a: np.ndarray) -> dict:
        """Weighted by the integer tokens n of the author: its total is the tokens T, the weight n / T sums to 1."""
        t = table(sh, n_a)
        for x in t.values():
            x["tokens"] = int(x.pop("total_weight"))
            x["total_weight"] = 1.0
        return t

    sets = {"lexicon_B": table(lexicon_shapes),
            "tokens_P": table(tokens_P, weight), "tokens_B": table(tokens_B, weight),
            "tokens_P_per_author": {a: per_author(tokens_P, author_tokens[a]) for a in counts},
            "tokens_B_per_author": {a: per_author(tokens_B, author_tokens[a]) for a in counts},
            "words": {v: {a: table(words_shapes[v][a]) for a in [*authors, "both"]} for v in VERSIONS}}

    tmp = Path(tempfile.mkdtemp(prefix=".paper_reference_range-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {key: cm.sha256(p) for key, p in path.items()},
                        "lexicon_sha256": lex.sha256, "dicdir_sha256": tagger.dicdir_sha256,
                        "library_sha256": cm.library_sha256(), **cm.versions("numpy", "fugashi", "ipadic"),
                        **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"R1_constants": {**CONSTANTS, "reference_tokens": values_rec["reference_tokens"]},
                          "R2_percentiles_of_the_mapped_lexicon_equal_calibrate_memolon": r2,
                          "R3_percentiles_of_values_json_recomputed": r3,
                          "R4_words_checked": r4,
                          "R5_entries_changed_by_the_clip_of_map_B": clip_count,
                          "R6_generator_and_rule": True},
               "rule_range": {q: list(v) for q, v in rule.items()},
               "sets": sets}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        with (tmp / RECORD).open("x") as fh:
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
