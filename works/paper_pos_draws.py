"""Sets of 120 drawn at random, 20 words of each author, from the words of the author passing the selection of parts of
speech of work 2 alone (before the filter on counts and works), beside the 120 words and the draws from all distinct
words, the filtered words and the candidates as recorded, under map B and the percentiles.

Usage:  uv run python works/paper_pos_draws.py   (refuses to overwrite)
Conditions: works/conditions/paper_pos_draws.json (committed before this script was written)
Input:  the 23 works of work 2 (as works/paper_filtered_draws.py), the lexicon of work 2 and the committed records named
        in the conditions (no drawing)
Output: results/paper_pos_draws/
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
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402
from atact import mapping  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_pos_draws.json"
OUT_DIR = ROOT / "results" / "paper_pos_draws"
SCRIPTS = {"words_script": "works/work2_words.py", "distinct_script": "works/paper_distinct_range.py",
           "filtered_script": "works/paper_filtered_draws.py", "reference_range_script": "works/paper_reference_range.py"}
COVERAGE_SCRIPT = "works/paper_coverage_ties_flat.py"  # not named by the conditions; checked here (after the review)
SHARED_CODE = ["works/work2_words.py", "works/_common.py", "src/pipeline.py", "src/japanese_text.py"]
CODE = ["works/paper_pos_draws.py", *SCRIPTS.values(), COVERAGE_SCRIPT, "checks/author_texture.py", "works/_common.py",
        "src/pipeline.py", "src/japanese_text.py"]
VERSIONS, QUANTITIES = ("B", "P"), ("f", "s")
SHARES = ("share_quartiles_percent", "share_min_max_percent")
RECORD = "pos_draws.json"
CONSTANTS = {"authors": 6, "words_per_author": 20, "tiles": 120, "reference_words": 18301, "replicates": 10000,
             "seed": 20261029}


def imported(name: str, rel: str):
    """A committed script imported as a module, checked to be that file."""
    mod = importlib.import_module(name)
    if Path(mod.__file__).resolve() != (ROOT / rel).resolve():
        raise SystemExit(f"the module imported is {mod.__file__}, not {rel}")
    return mod


def made_on(rec: dict, cond: Path, script: Path | None, what: str) -> None:
    """A record read was made on its conditions on a clean tree (by the script named, when the record names one)."""
    m = rec["meta"]
    if m["conditions_sha256"] != cm.sha256(cond) or m.get("git_dirty") is not False:
        raise SystemExit(f"{what} was not made on its conditions on a clean tree")
    if script is not None and m["script_sha256"] != cm.sha256(script):
        raise SystemExit(f"{what} was not made by {script.relative_to(ROOT)}")


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    if c["constants"] != CONSTANTS:
        raise SystemExit("the constants of the conditions differ from those of the script")
    if (c["random"]["replicates"], c["random"]["seed"]) != (CONSTANTS["replicates"], CONSTANTS["seed"]):
        raise SystemExit("the replicates or the seed of the conditions differ from its constants")
    if tuple(c["statistics"]["quantities"]) != QUANTITIES or not c["outputs"]["record"].startswith(RECORD + ":"):
        raise SystemExit("the quantities or the record of the conditions differ from those of the script")
    inp = c["inputs"]
    path = {k: ROOT / v["path"] for k, v in inp.items() if isinstance(v, dict)}
    if any(path[k] != ROOT / rel for k, rel in SCRIPTS.items()):
        raise SystemExit("the scripts named in the conditions are not those of this script")
    cm.check_file(path["values"], inp["values"]["sha256"], "values")
    w2 = json.loads(path["work2_conditions"].read_bytes())
    cm.same_as_head([COND, *path.values(), ROOT / w2["map"]["file"], *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2", "results/work2_words", "results/sketch_sp_color",
                          "results/paper_distinct_range", "results/paper_filtered_draws",
                          "results/paper_coverage_ties_flat", "results/paper_used_range", "results/paper_reference_range",
                          "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    ww = imported("work2_words", SCRIPTS["words_script"])  # select (and log_odds through it)
    dr = imported("paper_distinct_range", SCRIPTS["distinct_script"])  # draw and summary
    rr = imported("paper_reference_range", SCRIPTS["reference_range_script"])  # the reference and spread
    wc = json.loads(path["words_conditions"].read_bytes())
    rc = json.loads(path["reference_conditions"].read_bytes())
    words_rec = json.loads(path["words"].read_bytes())
    dist = json.loads(path["distinct_range"].read_bytes())
    fd = json.loads(path["filtered_draws"].read_bytes())
    cov = json.loads(path["coverage"].read_bytes())
    cal_raw = cm.check_file(path["calibration"], w2["map"]["sha256"], "the calibration (the map of work 2)")
    if json.loads(cal_raw)["meta"].get("git_dirty") is not False:
        raise SystemExit("calibrate_memolon.json was not computed on a clean tree")
    made_on(dist, path["distinct_conditions"], path["distinct_script"], "distinct_range.json")
    made_on(fd, path["filtered_conditions"], path["filtered_script"], "filtered_draws.json")
    made_on(cov, path["coverage_conditions"], ROOT / COVERAGE_SCRIPT, "coverage_ties_flat.json")
    if any(cov["meta"]["code_sha256"][p] != cm.sha256(ROOT / p) for p in SHARED_CODE):
        raise SystemExit("coverage_ties_flat.json was made with other code of the text step or of the selection")
    made_on(words_rec, path["words_conditions"], path["words_script"], "words.json")
    shared = {"distinct_range.json": (dist, ("work2_conditions", "work2_result", "words_conditions", "reference_conditions",
                                             "values", "calibration", "reference_range_script")),
              "filtered_draws.json": (fd, ("work2_conditions", "work2_result", "words_conditions", "words",
                                           "words_script", "reference_conditions", "values", "calibration",
                                           "distinct_conditions", "distinct_range", "distinct_script",
                                           "reference_range_script")),
              "coverage_ties_flat.json": (cov, ("work2_conditions", "work2_result", "words_conditions", "words",
                                                "reference_conditions", "values", "calibration", "filtered_draws"))}
    for what, (rec, keys) in shared.items():
        if any(rec["meta"]["inputs_sha256"][key] != cm.sha256(path[key]) for key in keys):
            raise SystemExit(f"{what} was not made from the inputs named here as committed")
    if wc["work2_conditions_sha256"] != cm.sha256(path["work2_conditions"]):
        raise SystemExit("work2.json is not the one the words were selected on")
    if rc["inputs"]["work2_conditions"] != "works/conditions/work2.json" or rc["reference"]["select"] != "all":
        raise SystemExit("the reference of sketch_sp_color is not the one of these conditions")
    cm.check_generator(wc["generator"])  # G6
    if (any(m["library_sha256"] != cm.library_sha256() for m in (dist["meta"], fd["meta"]))
            or any(m["atact_commit"] != cm.installed_commit() for m in (dist["meta"], fd["meta"], words_rec["meta"]))):
        raise SystemExit("G6: a record read was made with another generator")
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    if any(cm.versions("numpy")["numpy"] != m["numpy"] for m in (dist["meta"], fd["meta"])):
        raise SystemExit("numpy is not the version of distinct_range.json and filtered_draws.json")
    tagger = japanese_text.Tagger()
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if (tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"] or lex.sha256 != w2["lexicon"]["sha256"]
            or any(m["lexicon_sha256"] != lex.sha256 or m["dicdir_sha256"] != tagger.dicdir_sha256
                   for m in (dist["meta"], fd["meta"], words_rec["meta"]))
            or cov["meta"]["japanese_lexicon_sha256"] != lex.sha256 or cov["meta"]["dicdir_sha256"] != tagger.dicdir_sha256):
        raise SystemExit("the lexicon or the MeCab dictionary differs from that of work 2 or of a record read")
    # G2: the selection of parts of speech is the one of work 2 and of these conditions
    if (w2["step"]["select"] != "pos" or w2["step"]["pos_allowed"] != "japanese_text.POS_ALLOWED"
            or {p: sorted(s) for p, s in japanese_text.POS_ALLOWED.items()} != c["pos_allowed"]):
        raise SystemExit("G2: the selection of parts of speech is not that of work2.json and of the conditions")
    vmap = calibration_map(cal_raw)
    rule = {"f": (mapping.F0, mapping.F0 + mapping.F1), "s": (mapping.S0, mapping.S0 + mapping.S1)}  # G6
    if any(rec["rule_range"] != {q: list(v) for q, v in rule.items()} for rec in (dist, fd)):
        raise SystemExit("G6: the range of the rule of a record read is not the installed one")
    width = {q: hi - lo for q, (lo, hi) in rule.items()}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    k = CONSTANTS

    # G2: the text step, per work: the words of work 2 (select pos) and the reference (select all) ----------------
    done = {w["work_id"]: w for ws in json.loads(path["work2_result"].read_text())["authors"].values() for w in ws}
    words, author_of, tokens = {}, {}, {}
    counts: dict[str, Counter] = {}
    allowed = {p: set(x) for p, x in c["pos_allowed"].items()}
    direct: dict[str, set[str]] = {}  # after the review: the pos forms read from the tokens with the set of the conditions
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
            raise SystemExit(f"G2: {w['work_id']}: the text step gives another result than results/work2")
        words[w["work_id"]] = Counter(x for p in t.points for x in p.matched)
        author_of[w["work_id"]] = w["author"]
        tokens[w["work_id"]] = sum(words[w["work_id"]].values())
        everything = japanese_text.text_path(t.sentences, lex, tagger, vmap, select=rc["reference"]["select"], order="word")
        counts.setdefault(w["author"], Counter()).update(x for p in everything for x in p.matched)
        direct.setdefault(w["author"], set()).update(
            x.surface for sent in t.sentences for x in tagger._t(sent)
            if x.feature[1] in allowed.get(x.feature[0], ()) and x.surface in lex.entries)
    g2 = len(words)

    # G1: the authors -----------------------------------------------------------------------------------------------
    authors = list(words_rec["authors"])
    if (len(authors) != k["authors"] or set(authors) != set(author_of.values()) or set(counts) != set(authors)
            or list(fd["sets"]["pool_sizes"]["filtered"]) != authors):
        raise SystemExit("G1: the authors are not the six of words.json")
    if any(len(words_rec["authors"][a]["words"]) != k["words_per_author"] for a in authors):
        raise SystemExit("G1: an author of words.json has not 20 words")

    # G3: the values (as paper_filtered_draws.py) -------------------------------------------------------------------
    values_rec = json.loads(path["values"].read_bytes())
    wts = rr.reference_weights(counts)
    ref_words = sorted(wts)  # by Unicode code point, as distinct_range.json
    added = dist["checks"]["added_after_review"]
    if (len(ref_words) != k["reference_words"]
            or hashlib.sha256("\n".join(ref_words).encode("utf-8")).hexdigest() != added["reference_words_sha256"]):
        raise SystemExit("G3: the reference words are not those of distinct_range.json")
    rv = np.array([lex.entries[w][0] for w in ref_words])
    ra = np.array([lex.entries[w][1] for w in ref_words])
    weight = np.array([wts[w]["author_equal"] for w in ref_words])
    for key, v in values_rec["values"].items():
        e = v["entry"]
        got = (rr.weighted_percentile(rv, weight, lex.entries[e][0]), rr.weighted_percentile(ra, weight, lex.entries[e][1]))
        if got != (v["p_valence_author_equal"], v["p_arousal_author_equal"]):
            raise SystemExit(f"G3: the percentiles of {key} recomputed differ from values.json")
    p_v = np.array([rr.weighted_percentile(rv, weight, x) for x in rv])
    p_a = np.array([rr.weighted_percentile(ra, weight, x) for x in ra])
    shapes = {"P": rr.shapes(2 * p_v - 1, 2 * p_a - 1),  # G6 inside shapes
              "B": rr.shapes(*(np.array([vmap(*lex.entries[w])[i] for w in ref_words]) for i in (0, 1)))}
    if any(hashlib.sha256(shapes[v][q].tobytes()).hexdigest() != added["values_sha256"][f"{v}_{q}"]
           for v in VERSIONS for q in QUANTITIES):
        raise SystemExit("G3: the f and s of the reference are not those of distinct_range.json")
    index = {w: i for i, w in enumerate(ref_words)}

    def table(idx: np.ndarray, v: str) -> dict:
        return {q: rr.spread(shapes[v][q][idx], width[q]) for q in QUANTITIES}

    own_idx = np.array([index[r["word"]] for a in authors for r in words_rec["authors"][a]["words"]])
    own = {v: table(own_idx, v) for v in VERSIONS}
    if own != fd["sets"]["words_120"]:
        raise SystemExit("G3: the 120 words computed here differ from words_120 of filtered_draws.json")
    g3 = {"percentiles_of_values_json": len(values_rec["values"]), "reference_words_sha256": 1,
          "values_sha256": len(VERSIONS) * len(QUANTITIES), "words_120_equal_filtered_draws": 1}

    # G4: the pools nest --------------------------------------------------------------------------------------------
    sel = wc["selection"]
    if not (sel["min_count"] >= 1 and sel["min_works"] >= 1):  # else the filtered words could hold an absent word
        raise SystemExit("the filter of the selection admits a word absent from the author")
    real = ww.log_odds
    try:  # select itself with every z set to 1, so that the sign passes every word: the filtered words
        ww.log_odds = lambda o, r, pr: {x: 1.0 for x in pr}
        unsigned = ww.select(words, author_of, tokens, sel["min_count"], sel["min_works"], sel["max_share"], 10**9)
    finally:
        ww.log_odds = real
    full = ww.select(words, author_of, tokens, sel["min_count"], sel["min_works"], sel["max_share"], 10**9)
    if list(unsigned) != authors or list(full) != authors:
        raise SystemExit("G1: the authors of select are not those of words.json in its order")
    def twenty_rec(a: str) -> list[str]:
        return [r["word"] for r in words_rec["authors"][a]["words"]]

    pos_words: dict[str, list[str]] = {}
    sizes = {"all": {}, "pos": {}, "filtered": {}, "candidates": {}}
    g4 = 0
    for a in authors:
        mine = sum((words[w] for w in words if author_of[w] == a), Counter())
        pos = sorted(x for x, n in mine.items() if n > 0)
        every = {x for x, n in counts[a].items() if n > 0}
        filtered = {d["word"] for d in unsigned[a]["words"]}
        twenty = [r["word"] for r in words_rec["authors"][a]["words"]]
        if set(pos) != direct[a]:
            raise SystemExit(f"G4: {a}: the pos pool is not the forms read from the tokens with the set of the conditions")
        cands = [d["word"] for d in full[a]["words"]]
        if (len(cands) != fd["sets"]["pool_sizes"]["candidates"][a] or len(cands) != words_rec["authors"][a]["candidates"]
                or not set(cands) <= filtered or cands[:k["words_per_author"]] != twenty_rec(a)):
            raise SystemExit(f"G4: {a}: the candidates computed here differ from the records or are not filtered")
        if (not set(pos) <= every or not filtered <= set(pos) or any(x not in index for x in pos)
                or not set(twenty) <= set(pos)):
            raise SystemExit(f"G4: {a}: the pools do not nest (filtered within pos within all, 20 words within pos)")
        if (len(filtered) != fd["sets"]["pool_sizes"]["filtered"][a]
                or len(pos) != cov["C_work2_coverage"]["authors"][a]["distinct_in_lexicon"]
                or len(every) != dist["checks"]["D4_pool_sizes"][a]):
            raise SystemExit(f"G4: {a}: the sizes of the pools differ from the records")
        if len(pos) < k["words_per_author"]:
            raise SystemExit(f"G1: {a}: the pos pool holds fewer than 20 words")
        pos_words[a] = pos
        sizes["all"][a], sizes["pos"][a], sizes["filtered"][a] = len(every), len(pos), len(filtered)
        sizes["candidates"][a] = len(cands)
        g4 += 1
    pools = [np.array([index[x] for x in pos_words[a]], dtype=np.int64) for a in authors]

    # G5: the draws, made twice -------------------------------------------------------------------------------------
    reps = dr.draw(k["seed"], k["replicates"], pools, k["words_per_author"])
    again = dr.draw(k["seed"], k["replicates"], pools, k["words_per_author"])
    if len(reps) != k["replicates"] or any(not np.array_equal(x, y) for x, y in zip(reps, again, strict=True)):
        raise SystemExit("G5: the draws made again differ or their number is not the constant")
    rng = np.random.default_rng(k["seed"])  # after the review: the draws as the conditions describe them, not through draw
    if not isinstance(rng.bit_generator, np.random.PCG64) or any(
            not np.array_equal(x, np.concatenate([pool[rng.choice(pool.size, size=k["words_per_author"], replace=False)]
                                                  for pool in pools])) for x in reps):
        raise SystemExit("G5: the draws differ from a PCG64 generator drawing 20 without replacement per author")
    for x in reps:
        parts = np.split(x, len(authors))
        if x.size != k["tiles"] or any(np.unique(q).size != k["words_per_author"] or not np.isin(q, pool).all()
                                       for q, pool in zip(parts, pools, strict=True)):
            raise SystemExit("G5: a replicate is not 20 distinct words of each author from its pool")
    draws: dict = {}
    for v in VERSIONS:
        rep_tables = [table(x, v) for x in reps]
        draws[f"draws_pos_{v}"] = {
            "summary": {q: {sh: dr.summary(np.array([t[q][sh] for t in rep_tables]), own[v][q][sh]) for sh in SHARES}
                        for q in QUANTITIES},
            "replicates": [{q: {sh: t[q][sh] for sh in SHARES} for q in QUANTITIES} for t in rep_tables]}

    beside = {"draws_all": {v: dist["sets"][f"draws_{v}"]["summary"] for v in VERSIONS},
              **{f"draws_{p}": {v: fd["sets"][f"draws_{p}_{v}"]["summary"] for v in VERSIONS}
                 for p in ("filtered", "candidates")}}
    sets = {"pool_sizes": sizes,
            "pos_each_word_once": {a: {v: table(x, v) for v in VERSIONS} for a, x in zip(authors, pools, strict=True)},
            **draws, "words_120": own, "beside_as_recorded": beside}

    tmp = Path(tempfile.mkdtemp(prefix=".paper_pos_draws-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {key: cm.sha256(p) for key, p in path.items()},
                        "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                        "lexicon_sha256": lex.sha256, "dicdir_sha256": tagger.dicdir_sha256,
                        "library_sha256": cm.library_sha256(), "atact_commit": cm.installed_commit(),
                        **cm.versions("numpy", "fugashi", "ipadic"), **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"G1_constants": CONSTANTS,
                          "G2_works_through_the_text_step_and_pos_allowed": g2,
                          "G3_values_json_keys_words_sha_values_sha_and_120_words_checked": g3,
                          "G4_authors_whose_pools_nest_with_the_sizes_recorded": g4,
                          "G5_replicates_drawn_twice_and_by_hand": len(reps),
                          "added_after_review": {
                              "pos_read_from_the_tokens_with_pos_allowed_of_the_conditions": g4,
                              "candidates_computed_by_select_equal_filtered_draws_and_words_json": g4,
                              "coverage_script_lexicon_dictionary_and_shared_code_checked": 1 + 2 + len(SHARED_CODE),
                              "pos_sha256": {a: hashlib.sha256("\n".join(pos_words[a]).encode("utf-8")).hexdigest()
                                             for a in authors},
                              "pool_sizes_computed_here_all_pos_filtered_candidates": g4},
                          "G6_generator_and_rule": {"rule": {q: list(v) for q, v in rule.items()},
                                                    "values_within_minus_1_to_1": {v: len(ref_words) for v in VERSIONS}}},
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
    print(json.dumps({"checks": res["checks"]["G1_constants"], "pool_sizes": sizes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
