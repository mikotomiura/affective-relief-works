"""The share of the range of f and s used by the distinct words of the 23 works of work 2, each once and unweighted (the
unit of the 120 words of the main images), under map B and the percentiles, and by sets of 120 drawn at random, 20
distinct words of each author, beside the 120 words.

Usage:  uv run python works/paper_distinct_range.py   (refuses to overwrite)
Conditions: works/conditions/paper_distinct_range.json (committed before this script was written)
Input:  the 23 works of work 2 (as works/paper_reference_range.py), the lexicon of work 2 and the committed records named
        in the conditions (no drawing)
Output: results/paper_distinct_range/
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
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

COND = ROOT / "works" / "conditions" / "paper_distinct_range.json"
OUT_DIR = ROOT / "results" / "paper_distinct_range"
REF_SCRIPT = "works/paper_reference_range.py"
WORDS_FILE = "results/work2_words/words.json"  # the 120 words of used_range.json (an input of its conditions)
CODE = ["works/paper_distinct_range.py", REF_SCRIPT, "works/_common.py", "src/pipeline.py", "src/japanese_text.py"]
VERSIONS, QUANTITIES = ("B", "P"), ("f", "s")
USED_KEY = {"B": "main_G0", "P": "main_P"}
RECORD = "distinct_range.json"
CONSTANTS = {"authors": 6, "words_per_author": 20, "tiles": 120, "works": 23, "lexicon_entries": 2003306,
             "reference_words": 18301, "replicates": 10000, "seed": 20261029}
SHARES = ("share_quartiles_percent", "share_min_max_percent")


def draw(seed: int, reps: int, pools: list[np.ndarray], k: int) -> list[np.ndarray]:
    """R replicates of the indices of the drawn words: for every pool in order, k indices drawn uniformly without
    replacement by one generator, concatenated (a word drawn from two pools is kept once for each)."""
    rng = np.random.default_rng(seed)
    return [np.concatenate([pool[rng.choice(pool.size, size=k, replace=False)] for pool in pools]) for _ in range(reps)]


def summary(shares: np.ndarray, own: float) -> dict:
    """Mean, median and quantiles 0.025 and 0.975 (linear) over the replicates, and the replicates at most own."""
    lo, hi = (float(x) for x in np.quantile(shares, [0.025, 0.975], method="linear"))
    return {"mean": float(shares.mean()), "median": float(np.median(shares)), "q025": lo, "q975": hi,
            "at_most_120_words": int((shares <= own).sum()), "of": int(shares.size)}


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    if c["constants"] != CONSTANTS:
        raise SystemExit("the constants of the conditions differ from those of the script")
    rnd = c["random"]
    if (rnd["replicates"], rnd["seed"]) != (CONSTANTS["replicates"], CONSTANTS["seed"]):
        raise SystemExit("the replicates or the seed of the conditions differ from its constants")
    if tuple(c["statistics"]["quantities"]) != QUANTITIES or not c["outputs"]["record"].startswith(RECORD + ":"):
        raise SystemExit("the quantities or the record of the conditions differ from those of the script")
    inp = c["inputs"]
    path = {k: ROOT / v["path"] for k, v in inp.items() if isinstance(v, dict)}
    if path["reference_range_script"] != ROOT / REF_SCRIPT:
        raise SystemExit(f"the script of the reference sets named is not {REF_SCRIPT}")
    cm.check_file(path["values"], inp["values"]["sha256"], "values")
    w2 = json.loads(path["work2_conditions"].read_bytes())
    cm.same_as_head([COND, *path.values(), ROOT / w2["map"]["file"], ROOT / WORDS_FILE, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "results/work2", "results/sketch_sp_color", "results/paper_reference_range",
                          "results/paper_used_range", "results/work2_words", "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    rr = importlib.import_module("paper_reference_range")  # the committed functions, not copies
    if Path(rr.__file__).resolve() != (ROOT / REF_SCRIPT).resolve():
        raise SystemExit(f"the module imported is {rr.__file__}, not {REF_SCRIPT}")
    wc = json.loads(path["words_conditions"].read_bytes())
    rc = json.loads(path["reference_conditions"].read_bytes())
    ref_rec = json.loads(path["reference_range"].read_bytes())
    used = json.loads(path["used_range"].read_bytes())
    cal_raw = cm.check_file(path["calibration"], w2["map"]["sha256"], "the calibration (the map of work 2)")
    if json.loads(cal_raw)["meta"].get("git_dirty") is not False:
        raise SystemExit("calibrate_memolon.json was not computed on a clean tree")
    for rec, cond_key, what in ((ref_rec, "reference_range_conditions", "reference_range.json"),
                                (used, "used_range_conditions", "used_range.json")):
        if rec["meta"]["conditions_sha256"] != cm.sha256(path[cond_key]) or rec["meta"].get("git_dirty") is not False:
            raise SystemExit(f"{what} was not made on its conditions on a clean tree")
    for rec, what, shared in ((ref_rec, "reference_range.json", ("work2_conditions", "work2_result", "words_conditions",
                                                                 "reference_conditions", "values", "calibration")),
                              (used, "used_range.json", ("work2_conditions", "words_conditions", "reference_conditions",
                                                         "values", "calibration", "reference_range_conditions",
                                                         "reference_range"))):
        # the same inputs as these conditions (after the review: the provenance of both records, not of one)
        if any(rec["meta"]["inputs_sha256"][key] != cm.sha256(path[key]) for key in shared):
            raise SystemExit(f"{what} was not made from the inputs named here as committed")
        if rec["meta"]["atact_commit"] != cm.installed_commit():
            raise SystemExit(f"D6: {what} was made with another commit of the generator")
    if ref_rec["meta"]["script_sha256"] != cm.sha256(ROOT / REF_SCRIPT):
        raise SystemExit(f"reference_range.json was not made by {REF_SCRIPT}")
    if wc["work2_conditions_sha256"] != cm.sha256(path["work2_conditions"]):
        raise SystemExit("work2.json is not the one the words were selected on")
    if rc["inputs"]["work2_conditions"] != "works/conditions/work2.json" or rc["reference"]["select"] != "all":
        raise SystemExit("the reference of sketch_sp_color is not the one of these conditions")
    cm.check_generator(wc["generator"])  # D6
    if ref_rec["meta"]["library_sha256"] != cm.library_sha256() or used["meta"]["library_sha256"] != cm.library_sha256():
        raise SystemExit("D6: a record read was made with another generator")
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of work 2")
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if lex.sha256 != w2["lexicon"]["sha256"] or any(rec["meta"]["lexicon_sha256"] != lex.sha256 for rec in (ref_rec, used)):
        raise SystemExit("the lexicon differs from that of work 2 or of a record read")
    if any(rec["meta"]["dicdir_sha256"] != tagger.dicdir_sha256 for rec in (ref_rec, used)):
        raise SystemExit("the MeCab dictionary differs from that of a record read")
    vmap = calibration_map(cal_raw)
    rule = {"f": (mapping.F0, mapping.F0 + mapping.F1), "s": (mapping.S0, mapping.S0 + mapping.S1)}  # D6
    if not all(np.allclose(rule[q], want, rtol=0.0, atol=1e-12) for q, want in (("f", (3.0, 7.0)), ("s", (0.06, 0.30)))):
        raise SystemExit(f"D6: the range of the rule {rule} is not the one of the conditions")
    if any(rec["rule_range"] != {q: list(v) for q, v in rule.items()} for rec in (ref_rec, used)):
        raise SystemExit("D6: the range of the rule of a record read is not the installed one")
    width = {q: hi - lo for q, (lo, hi) in rule.items()}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # D1: the constants ---------------------------------------------------
    k = CONSTANTS
    authors = [a for a in used["sets"]["main_G0"] if a != "pooled"]
    if (len(w2["works"]) != k["works"] or {w["author"] for w in w2["works"]} != set(authors)
            or len(authors) != k["authors"] or len(lex.entries) != k["lexicon_entries"]):
        raise SystemExit("D1: the works, authors or entries of the lexicon are not the constants")
    for v in VERSIONS:
        s = used["sets"][USED_KEY[v]]
        if ([a for a in s if a != "pooled"] != authors
                or any(s[a][q]["n"] != k["words_per_author"] for a in authors for q in QUANTITIES)
                or any(s["pooled"][q]["n"] != k["tiles"] for q in QUANTITIES)):
            raise SystemExit(f"D1: the 120 words of {USED_KEY[v]} in used_range.json are not the constants")

    # D2: the reference of the 23 works (as paper_reference_range.py) -----
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
            raise SystemExit(f"D2: {w['work_id']}: the text step gives another result than results/work2")
        everything = japanese_text.text_path(t.sentences, lex, tagger, vmap, select=rc["reference"]["select"], order="word")
        counts.setdefault(w["author"], Counter()).update(x for p in everything for x in p.matched)
    values_rec = json.loads(path["values"].read_bytes())
    if set(counts) != set(authors) or {a: sum(n.values()) for a, n in counts.items()} != values_rec["reference_tokens"]:
        raise SystemExit("D1: the tokens per author differ from those of values.json")
    wts = rr.reference_weights(counts)
    ref_words = sorted(wts)  # by Unicode code point (Python sorted); the order the pools and the draws index
    if len(ref_words) != k["reference_words"] or values_rec["reference_words"] != k["reference_words"]:
        raise SystemExit(f"D1: {len(ref_words)} reference words, not {k['reference_words']}")
    rv = np.array([lex.entries[w][0] for w in ref_words])
    ra = np.array([lex.entries[w][1] for w in ref_words])
    weight = np.array([wts[w]["author_equal"] for w in ref_words])
    d2 = 0
    for key, v in values_rec["values"].items():
        e = v["entry"]
        got = (rr.weighted_percentile(rv, weight, lex.entries[e][0]), rr.weighted_percentile(ra, weight, lex.entries[e][1]))
        if got != (v["p_valence_author_equal"], v["p_arousal_author_equal"]):
            raise SystemExit(f"D2: the percentiles of {key} recomputed differ from values.json")
        d2 += 1
    p_v = np.array([rr.weighted_percentile(rv, weight, x) for x in rv])
    p_a = np.array([rr.weighted_percentile(ra, weight, x) for x in ra])
    shapes = {"P": rr.shapes(2 * p_v - 1, 2 * p_a - 1),  # D6 inside shapes
              "B": rr.shapes(*(np.array([vmap(*lex.entries[w])[i] for w in ref_words]) for i in (0, 1)))}

    # D3: the weighted sets of reference_range.json, recomputed -------------
    for v, name in (("B", "tokens_B"), ("P", "tokens_P")):
        again = {q: rr.spread(shapes[v][q], width[q], weight) for q in QUANTITIES}
        if again != ref_rec["sets"][name]:
            raise SystemExit(f"D3: {name} recomputed differs from reference_range.json")
    d3 = 2 * len(QUANTITIES)

    # D4: the pools of the draws ------------------------------------------
    index = {w: i for i, w in enumerate(ref_words)}
    pools = [np.array(sorted(index[w] for w, n in counts[a].items() if n > 0), dtype=np.int64) for a in authors]
    if (set().union(*(set(p.tolist()) for p in pools)) != set(range(len(ref_words)))
            or any(p.size < k["words_per_author"] for p in pools)):
        raise SystemExit("D4: the pools of the authors do not cover the reference words or one holds fewer than 20")

    def table(idx: np.ndarray | None, v: str) -> dict:
        sel = slice(None) if idx is None else idx
        return {q: rr.spread(shapes[v][q][sel], width[q]) for q in QUANTITIES}

    # D5: the draws, made twice ---------------------------------------------
    reps = draw(k["seed"], k["replicates"], pools, k["words_per_author"])
    again = draw(k["seed"], k["replicates"], pools, k["words_per_author"])
    if len(reps) != k["replicates"] or any(not np.array_equal(x, y) for x, y in zip(reps, again, strict=True)):
        raise SystemExit("D5: the draws made again differ or their number is not the constant")
    for x in reps:
        parts = np.split(x, len(authors))
        if x.size != k["tiles"] or any(np.unique(p).size != k["words_per_author"] or not np.isin(p, pool).all()
                                       for p, pool in zip(parts, pools, strict=True)):
            raise SystemExit("D5: a replicate is not 20 distinct words of each author")
    # beyond the checks of the conditions (after the review): the 120 words, computed here by the same values and the
    # same table, equal the pooled sets of used_range.json, and each is in the pool of its author (so own and the draws
    # come by one path)
    words_rec = json.loads((ROOT / WORDS_FILE).read_bytes())
    if used["meta"]["inputs_sha256"]["words"] != cm.sha256(ROOT / WORDS_FILE):
        raise SystemExit(f"{WORDS_FILE} is not the one of used_range.json")
    own_idx = []
    for a, pool in zip(authors, pools, strict=True):
        ws = [r["word"] for r in words_rec["authors"][a]["words"]]
        if len(ws) != k["words_per_author"] or any(w not in index or index[w] not in set(pool.tolist()) for w in ws):
            raise SystemExit(f"{a}: a word of {WORDS_FILE} is not in the pool of its author")
        own_idx += [index[w] for w in ws]
    for v in VERSIONS:
        if table(np.array(own_idx), v) != used["sets"][USED_KEY[v]]["pooled"]:
            raise SystemExit(f"the 120 words computed here under {v} differ from used_range.json")
    draws: dict = {}
    for v in VERSIONS:
        rep_tables = [table(x, v) for x in reps]
        rep_again = [table(x, v) for x in again]
        if rep_tables != rep_again:
            raise SystemExit(f"D5: the shares of the draws made again differ under {v}")
        own = used["sets"][USED_KEY[v]]["pooled"]
        draws[v] = {"summary": {q: {sh: summary(np.array([t[q][sh] for t in rep_tables]), own[q][sh]) for sh in SHARES}
                                for q in QUANTITIES},
                    "replicates": [{q: {sh: t[q][sh] for sh in SHARES} for q in QUANTITIES} for t in rep_tables]}

    sets = {"distinct_B": table(None, "B"), "distinct_P": table(None, "P"),
            "distinct_per_author": {a: {v: table(pool, v) for v in VERSIONS} for a, pool in zip(authors, pools, strict=True)},
            "draws_B": draws["B"], "draws_P": draws["P"],
            "words_120": {v: used["sets"][USED_KEY[v]]["pooled"] for v in VERSIONS}}

    tmp = Path(tempfile.mkdtemp(prefix=".paper_distinct_range-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {key: cm.sha256(p) for key, p in path.items()},
                        "lexicon_sha256": lex.sha256, "dicdir_sha256": tagger.dicdir_sha256,
                        "library_sha256": cm.library_sha256(), **cm.versions("numpy", "fugashi", "ipadic"),
                        **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"D1_constants": {**CONSTANTS, "reference_tokens": values_rec["reference_tokens"]},
                          "D2_percentiles_of_values_json_recomputed": d2,
                          "D3_weighted_tables_equal_reference_range": d3,
                          "D4_pool_sizes": {a: int(p.size) for a, p in zip(authors, pools, strict=True)},
                          "D5_replicates_drawn_twice": len(reps),
                          "D6_generator_and_rule": {"rule": {q: list(v) for q, v in rule.items()},
                                                    "values_within_minus_1_to_1": {v: len(ref_words) for v in VERSIONS}},
                          "added_after_review": {
                              "words_120_computed_here_equal_used_range": 2,
                              "D3_meaning": "the weighted summaries (n, total weight, minimum, quartiles, maximum, shares) of "
                                            "tokens_B and tokens_P equal the record; the words themselves rest on D1 and D2",
                              "D4_note": "the pools are built from the counts of the reference, so their union is the "
                                         "reference words by construction; the check that bites is at least 20 words",
                              "D5_note": "the shares drawn again follow from the words drawn again",
                              "order_of_the_reference_words": "Unicode code point (Python sorted)",
                              "reference_words_sha256": hashlib.sha256("\n".join(ref_words).encode("utf-8")).hexdigest(),
                              "values_sha256": {f"{v}_{q}": hashlib.sha256(shapes[v][q].tobytes()).hexdigest()
                                                for v in VERSIONS for q in QUANTITIES}}},
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
