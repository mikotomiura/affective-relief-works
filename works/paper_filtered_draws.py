"""Sets of 120 drawn at random, 20 words of each author, from the words passing the filter of the selection of work 2
and from its candidates (the filter and z > 0), beside the 120 words (the 20 of largest z) and the draws from all
distinct words, under map B and the percentiles: at which step of the selection the spread of the 120 words narrows.

Usage:  uv run python works/paper_filtered_draws.py   (refuses to overwrite)
Conditions: works/conditions/paper_filtered_draws.json (committed before this script was written)
Input:  the 23 works of work 2 (as works/paper_distinct_range.py), the lexicon of work 2 and the committed records named
        in the conditions (no drawing)
Output: results/paper_filtered_draws/
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

COND = ROOT / "works" / "conditions" / "paper_filtered_draws.json"
OUT_DIR = ROOT / "results" / "paper_filtered_draws"
SCRIPTS = {"words_script": "works/work2_words.py", "distinct_script": "works/paper_distinct_range.py",
           "reference_range_script": "works/paper_reference_range.py"}
CODE = ["works/paper_filtered_draws.py", *SCRIPTS.values(), "checks/author_texture.py", "works/_common.py",
        "src/pipeline.py", "src/japanese_text.py"]
VERSIONS, QUANTITIES, POOLS = ("B", "P"), ("f", "s"), ("filtered", "candidates")
SHARES = ("share_quartiles_percent", "share_min_max_percent")
RECORD = "filtered_draws.json"
CONSTANTS = {"authors": 6, "words_per_author": 20, "tiles": 120, "reference_words": 18301, "replicates": 10000,
             "seed": 20261029, "candidates": {"夏目漱石": 325}}


def imported(name: str, rel: str):
    """A committed script imported as a module, checked to be that file."""
    mod = importlib.import_module(name)
    if Path(mod.__file__).resolve() != (ROOT / rel).resolve():
        raise SystemExit(f"the module imported is {mod.__file__}, not {rel}")
    return mod


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
                          "results/paper_distinct_range", "results/paper_used_range", "results/paper_reference_range",
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
    cal_raw = cm.check_file(path["calibration"], w2["map"]["sha256"], "the calibration (the map of work 2)")
    if json.loads(cal_raw)["meta"].get("git_dirty") is not False:
        raise SystemExit("calibrate_memolon.json was not computed on a clean tree")
    dm = dist["meta"]
    if (dm["conditions_sha256"] != cm.sha256(path["distinct_conditions"]) or dm.get("git_dirty") is not False
            or dm["script_sha256"] != cm.sha256(path["distinct_script"])):
        raise SystemExit("distinct_range.json was not made on its conditions on a clean tree by the script named")
    shared = ("work2_conditions", "work2_result", "words_conditions", "reference_conditions", "values", "calibration",
              "reference_range_script")
    if any(dm["inputs_sha256"][k] != cm.sha256(path[k]) for k in shared):
        raise SystemExit("distinct_range.json was not made from the inputs named here as committed")
    # words.json: made on its conditions by the script named, and the one behind distinct_range.json (through the record
    # of used_range.json, an input of it, which names words.json among its inputs)
    wm = words_rec["meta"]
    used_rel = json.loads(path["distinct_conditions"].read_bytes())["inputs"]["used_range"]["path"]
    used = json.loads(cm.check_file(ROOT / used_rel, dm["inputs_sha256"]["used_range"], "used_range.json"))
    cm.same_as_head([ROOT / used_rel])
    if (wm["conditions_sha256"] != cm.sha256(path["words_conditions"]) or wm.get("git_dirty") is not False
            or wm["script_sha256"] != cm.sha256(path["words_script"])
            or used["meta"]["inputs_sha256"]["words"] != cm.sha256(path["words"])):
        raise SystemExit("words.json is not the record of its conditions and script, or not the one of distinct_range.json")
    if wc["work2_conditions_sha256"] != cm.sha256(path["work2_conditions"]):
        raise SystemExit("work2.json is not the one the words were selected on")
    if rc["inputs"]["work2_conditions"] != "works/conditions/work2.json" or rc["reference"]["select"] != "all":
        raise SystemExit("the reference of sketch_sp_color is not the one of these conditions")
    cm.check_generator(wc["generator"])  # F6
    if (dm["library_sha256"] != cm.library_sha256() or dm["atact_commit"] != cm.installed_commit()
            or wm["atact_commit"] != cm.installed_commit()):  # words.json records the commit only; its library is that of the generator of its conditions (check_generator)
        raise SystemExit("F6: a record read was made with another generator")
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    if cm.versions("numpy")["numpy"] != dm["numpy"]:  # the draws and the quantiles of distinct_range.json
        raise SystemExit("numpy is not the version of distinct_range.json")
    tagger = japanese_text.Tagger()
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if (tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"] or lex.sha256 != w2["lexicon"]["sha256"]
            or any(m["lexicon_sha256"] != lex.sha256 or m["dicdir_sha256"] != tagger.dicdir_sha256 for m in (dm, wm))):
        raise SystemExit("the lexicon or the MeCab dictionary differs from that of work 2 or of a record read")
    vmap = calibration_map(cal_raw)
    rule = {"f": (mapping.F0, mapping.F0 + mapping.F1), "s": (mapping.S0, mapping.S0 + mapping.S1)}  # F6
    if dist["rule_range"] != {q: list(v) for q, v in rule.items()}:
        raise SystemExit("F6: the range of the rule of distinct_range.json is not the installed one")
    width = {q: hi - lo for q, (lo, hi) in rule.items()}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    k = CONSTANTS

    # the text step, per work: the words of the selection and the reference (as work2_words.py and the distinct range)
    done = {w["work_id"]: w for ws in json.loads(path["work2_result"].read_text())["authors"].values() for w in ws}
    words, author_of, tokens, titles = {}, {}, {}, {}
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
            raise SystemExit(f"{w['work_id']}: the text step gives another result than results/work2")
        words[w["work_id"]] = Counter(x for p in t.points for x in p.matched)
        author_of[w["work_id"]], titles[w["work_id"]] = w["author"], w["title"]
        tokens[w["work_id"]] = sum(words[w["work_id"]].values())
        everything = japanese_text.text_path(t.sentences, lex, tagger, vmap, select=rc["reference"]["select"], order="word")
        counts.setdefault(w["author"], Counter()).update(x for p in everything for x in p.matched)

    # F1 and F2: the selection again ----------------------------------------
    authors = list(words_rec["authors"])
    sel = wc["selection"]
    if not (sel["min_count"] >= 1 and sel["min_works"] >= 1):  # else the words of the author would not be all the words
        raise SystemExit("the filter of the selection admits a word absent from the author")
    chosen = ww.select(words, author_of, tokens, sel["min_count"], sel["min_works"], sel["max_share"], sel["n_words"])
    if (len(authors) != k["authors"] or list(chosen) != authors
            or any(chosen[a]["candidates"] != words_rec["authors"][a]["candidates"] for a in authors)
            or any(words_rec["authors"][a]["candidates"] != n for a, n in k["candidates"].items())):
        raise SystemExit("F1: the authors or the numbers of candidates are not those of words.json and the constants")
    f2 = 0
    for a in authors:
        rec_rows = words_rec["authors"][a]["words"]
        if len(rec_rows) != k["words_per_author"] or len(chosen[a]["words"]) != k["words_per_author"]:
            raise SystemExit(f"F1: {a}: not 20 words")
        for got, want in zip(chosen[a]["words"], rec_rows, strict=True):
            mine = {"word": got["word"], "z": got["z"], "share": got["share"],
                    "counts": {titles[w]: n for w, n in got["counts"].items()}}  # words.json names the works by title
            if mine != {x: want[x] for x in ("word", "z", "share", "counts")}:
                raise SystemExit(f"F2: {a}: the selection again differs from words.json at {want['word']}")
            f2 += 1
    # the filter and the sign, as select applies them, kept apart
    by_author = {a: sum((words[w] for w in words if author_of[w] == a), Counter()) for a in authors}
    total = sum(by_author.values(), Counter())
    n_total = sum(total.values())
    prior = {w: ww.PRIOR_TOTAL * n / n_total for w, n in total.items()}
    pools_words: dict[str, dict[str, list[str]]] = {p: {} for p in POOLS}
    every_candidate: dict[str, list[str]] = {}
    for a in authors:
        own_works = [w for w in words if author_of[w] == a]
        rest = sum((by_author[b] for b in authors if b != a), Counter())
        z = ww.log_odds(by_author[a], rest, prior)
        filtered, cands = [], []
        for word in by_author[a]:  # the words of the author (z of a word not of the author is not a candidate)
            cnt = {w: words[w][word] for w in own_works}
            if sum(x >= sel["min_count"] for x in cnt.values()) < sel["min_works"]:
                continue
            rates = [cnt[w] / tokens[w] for w in own_works]
            if max(rates) / sum(rates) > sel["max_share"] + 1e-12:
                continue
            filtered.append(word)
            if z[word] > 0:
                cands.append(word)
        order = sorted(cands, key=lambda x: (-z[x], x))
        if len(cands) != chosen[a]["candidates"] or order[:k["words_per_author"]] != [r["word"] for r in
                                                                                  words_rec["authors"][a]["words"]]:
            raise SystemExit(f"F2: {a}: the candidates computed here are not those of select")
        pools_words["filtered"][a], pools_words["candidates"][a] = sorted(filtered), sorted(cands)
        every_candidate[a] = order
    # after the review: both pools checked against select itself, every word: the candidates as select returns them all,
    # and the filtered words as select returns them when every z is 1 (the sign then passes every word)
    full = ww.select(words, author_of, tokens, sel["min_count"], sel["min_works"], sel["max_share"], 10**9)
    real = ww.log_odds
    try:
        ww.log_odds = lambda o, r, pr: {w: 1.0 for w in pr}
        unsigned = ww.select(words, author_of, tokens, sel["min_count"], sel["min_works"], sel["max_share"], 10**9)
    finally:
        ww.log_odds = real
    for a in authors:
        if [d["word"] for d in full[a]["words"]] != every_candidate[a]:
            raise SystemExit(f"F2: {a}: the candidates computed here are not every candidate of select, in its order")
        if sorted(d["word"] for d in unsigned[a]["words"]) != pools_words["filtered"][a]:
            raise SystemExit(f"F2: {a}: the filtered words computed here are not those of select without the sign")

    # F3: the values (as paper_distinct_range.py) ----------------------------
    values_rec = json.loads(path["values"].read_bytes())
    wts = rr.reference_weights(counts)
    ref_words = sorted(wts)  # by Unicode code point, as distinct_range.json
    added = dist["checks"]["added_after_review"]
    if (len(ref_words) != k["reference_words"]
            or hashlib.sha256("\n".join(ref_words).encode("utf-8")).hexdigest() != added["reference_words_sha256"]):
        raise SystemExit("F3: the reference words are not those of distinct_range.json")
    rv = np.array([lex.entries[w][0] for w in ref_words])
    ra = np.array([lex.entries[w][1] for w in ref_words])
    weight = np.array([wts[w]["author_equal"] for w in ref_words])
    for key, v in values_rec["values"].items():
        e = v["entry"]
        got = (rr.weighted_percentile(rv, weight, lex.entries[e][0]), rr.weighted_percentile(ra, weight, lex.entries[e][1]))
        if got != (v["p_valence_author_equal"], v["p_arousal_author_equal"]):
            raise SystemExit(f"F3: the percentiles of {key} recomputed differ from values.json")
    p_v = np.array([rr.weighted_percentile(rv, weight, x) for x in rv])
    p_a = np.array([rr.weighted_percentile(ra, weight, x) for x in ra])
    shapes = {"P": rr.shapes(2 * p_v - 1, 2 * p_a - 1),  # F6 inside shapes
              "B": rr.shapes(*(np.array([vmap(*lex.entries[w])[i] for w in ref_words]) for i in (0, 1)))}
    if any(hashlib.sha256(shapes[v][q].tobytes()).hexdigest() != added["values_sha256"][f"{v}_{q}"]
           for v in VERSIONS for q in QUANTITIES):
        raise SystemExit("F3: the f and s of the reference are not those of distinct_range.json")
    index = {w: i for i, w in enumerate(ref_words)}

    # F4: the pools ----------------------------------------------------------
    pools: dict[str, list[np.ndarray]] = {}
    for p in POOLS:
        if any(w not in index for a in authors for w in pools_words[p][a]):
            raise SystemExit(f"F4: a word of {p} is not a reference word")
        pools[p] = [np.array([index[w] for w in pools_words[p][a]], dtype=np.int64) for a in authors]
    if any(not set(pools_words["candidates"][a]) <= set(pools_words["filtered"][a]) for a in authors):
        raise SystemExit("F4: the candidates are not a subset of filtered")
    if any(x.size < k["words_per_author"] for p in POOLS for x in pools[p]):
        raise SystemExit("F1: a pool holds fewer than 20 words")

    def table(idx: np.ndarray, v: str) -> dict:
        return {q: rr.spread(shapes[v][q][idx], width[q]) for q in QUANTITIES}

    f3 = len(values_rec["values"]) + 1 + len(VERSIONS) * len(QUANTITIES)
    own_idx = np.array([index[r["word"]] for a in authors for r in words_rec["authors"][a]["words"]])
    own = {v: table(own_idx, v) for v in VERSIONS}
    if own != dist["sets"]["words_120"]:
        raise SystemExit("F3: the 120 words computed here differ from words_120 of distinct_range.json")
    f3 += len(VERSIONS)

    # F5: the draws, made twice ----------------------------------------------
    draws: dict = {}
    for p in POOLS:
        reps = dr.draw(k["seed"], k["replicates"], pools[p], k["words_per_author"])
        again = dr.draw(k["seed"], k["replicates"], pools[p], k["words_per_author"])
        if len(reps) != k["replicates"] or any(not np.array_equal(x, y) for x, y in zip(reps, again, strict=True)):
            raise SystemExit(f"F5: the draws of {p} made again differ or their number is not the constant")
        for x in reps:
            parts = np.split(x, len(authors))
            if x.size != k["tiles"] or any(np.unique(q).size != k["words_per_author"] or not np.isin(q, pool).all()
                                           for q, pool in zip(parts, pools[p], strict=True)):
                raise SystemExit(f"F5: a replicate of {p} is not 20 distinct words of each author from its pool")
        for v in VERSIONS:
            rep_tables = [table(x, v) for x in reps]
            draws[f"draws_{p}_{v}"] = {
                "summary": {q: {sh: dr.summary(np.array([t[q][sh] for t in rep_tables]), own[v][q][sh]) for sh in SHARES}
                            for q in QUANTITIES},
                "replicates": [{q: {sh: t[q][sh] for sh in SHARES} for q in QUANTITIES} for t in rep_tables]}

    sets = {"pool_sizes": {p: {a: int(x.size) for a, x in zip(authors, pools[p], strict=True)} for p in POOLS},
            "pools_each_word_once": {p: {a: {v: table(x, v) for v in VERSIONS} for a, x in zip(authors, pools[p], strict=True)}
                                     for p in POOLS},
            **draws, "words_120": own,
            "draws_all": {v: dist["sets"][f"draws_{v}"]["summary"] for v in VERSIONS}}

    tmp = Path(tempfile.mkdtemp(prefix=".paper_filtered_draws-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {**{key: cm.sha256(p) for key, p in path.items()},
                                          "used_range": cm.sha256(ROOT / used_rel)},
                        "lexicon_sha256": lex.sha256, "dicdir_sha256": tagger.dicdir_sha256,
                        "library_sha256": cm.library_sha256(), **cm.versions("numpy", "fugashi", "ipadic"),
                        **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"F1_constants": CONSTANTS,
                          "F2_words_of_the_selection_again": f2,
                          "F3_values_json_keys_words_sha_values_sha_and_120_words_checked": f3,
                          "F4_pools": {p: {a: int(x.size) for a, x in zip(authors, pools[p], strict=True)} for p in POOLS},
                          "F4_note": "the candidates are a subset of filtered by construction; both pools are checked "
                                     "against select itself in F2 (after the review)",
                          "F5_replicates_drawn_twice": {p: k["replicates"] for p in POOLS},
                          "F6_generator_and_rule": {"rule": {q: list(v) for q, v in rule.items()},
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
    print(json.dumps(res["checks"]["F1_constants"], ensure_ascii=False), "F2", f2)


if __name__ == "__main__":
    main()
