"""The share of the range of f and s used by the 120 words of the main images of work 2 (under map B and under the
author-equal percentiles) and by the 104 sentences of work 1, minimum to maximum and quartiles, next to what they are
compared with (the same words under the other version, the reference sets of work 2 as recorded, the NRC lexicon).

Usage:  uv run python works/paper_used_range.py   (refuses to overwrite)
Conditions: works/conditions/paper_used_range.json (committed before this script was written)
Input:  the 23 works of work 2 (as works/paper_reference_range.py), the lexicons of works 1 and 2 and the committed
        records named in the conditions (no drawing)
Output: results/paper_used_range/
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
from atact.lexicon import load_lexicon  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_used_range.json"
OUT_DIR = ROOT / "results" / "paper_used_range"
CODE = ["works/paper_used_range.py", "works/_common.py", "src/pipeline.py", "src/japanese_text.py"]
QUANTITIES = ("f", "s")
RECORD = "used_range.json"
CONSTANTS = {"authors": 6, "words_per_author": 20, "tiles": 120, "distinct_words": 117, "works": 23,
             "reference_words": 18301, "work1_sentences": 104}


def weighted_percentile(values: np.ndarray, weights: np.ndarray, x: float) -> float:
    """Copied from works/sketch_sp_color.py: weight of the values below x plus half the weight of those equal to x, over
    the total weight."""
    total = weights.sum()
    if not total > 0:
        raise ValueError("no weight")
    return float((weights[values < x].sum() + 0.5 * weights[values == x].sum()) / total)


def reference_weights(counts: dict[str, Counter]) -> dict[str, float]:
    """As works/sketch_sp_color.py (the author-equal weight only): per word the sum over authors of n / T."""
    out: dict[str, float] = {}
    for a, cnt in counts.items():
        T = sum(cnt.values())
        if T == 0:
            raise ValueError(f"{a}: no token")
        for w, n in cnt.items():
            out[w] = out.get(w, 0.0) + n / T
    return out


def spread(xs: np.ndarray, width: float) -> dict:
    """Minimum, quartiles (linear), maximum and the two shares of the width of the range of the rule."""
    q1, q3 = (float(x) for x in np.quantile(xs, [0.25, 0.75], method="linear"))
    lo, hi = float(xs.min()), float(xs.max())
    return {"n": int(xs.size), "min": lo, "q1": q1, "q3": q3, "max": hi,
            "share_min_max_percent": 100 * (hi - lo) / width, "share_quartiles_percent": 100 * (q3 - q1) / width}


def shapes(V: np.ndarray, A: np.ndarray) -> dict[str, np.ndarray]:
    """f and s of atact.mapping.params, after checking that params clips nothing (U6)."""
    V, A = np.asarray(V, dtype=float), np.asarray(A, dtype=float)
    if not (np.all((V >= -1.0) & (V <= 1.0)) and np.all((A >= -1.0) & (A <= 1.0))):
        raise SystemExit("U6: a V or A outside [-1, 1]")
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
    c1 = json.loads(path["work1_conditions"].read_bytes())
    cm.same_as_head([COND, *path.values(), ROOT / w2["map"]["file"], ROOT / "results/work2/work2.json",
                  *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "results/work2", "results/work2_words", "results/work2_sp_color",
                          "results/sketch_sp_color", "results/paper_reference_range", "results/work1",
                          "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    wc = json.loads(path["words_conditions"].read_bytes())
    mc = json.loads(path["main_conditions"].read_bytes())
    rc = json.loads(path["reference_conditions"].read_bytes())
    words = json.loads(path["words"].read_bytes())
    main_rec = json.loads(path["main_record"].read_bytes())
    rr = json.loads(path["reference_range"].read_bytes())
    w1 = json.loads(path["work1"].read_bytes())
    cal_raw = cm.check_file(path["calibration"], w2["map"]["sha256"], "the calibration (the map of work 2)")
    cal = json.loads(cal_raw)
    if cal["meta"].get("git_dirty") is not False:
        raise SystemExit("calibrate_memolon.json was not computed on a clean tree")
    # each record made on its conditions, on a clean tree
    for name, rec, cond in (("words.json", words, "words_conditions"), ("work2_sp_color.json", main_rec, "main_conditions"),
                            ("reference_range.json", rr, "reference_range_conditions"), ("work1.json", w1, "work1_conditions")):
        if rec["meta"]["conditions_sha256"] != cm.sha256(path[cond]) or rec["meta"].get("git_dirty") is not False:
            raise SystemExit(f"{name} was not made on its conditions on a clean tree")
    if (wc["work2_conditions_sha256"] != cm.sha256(path["work2_conditions"])
            or main_rec["meta"]["words_sha256"] != cm.sha256(path["words"])
            or main_rec["meta"]["sketch_values_sha256"] != inp["values"]["sha256"]
            or rr["meta"]["inputs_sha256"]["values"] != inp["values"]["sha256"]):
        raise SystemExit("the records of work 2 are not the chain of work2.json, words.json and values.json")
    for name, cond in (("work2_sp_color", mc), ("sketch_sp_color", rc)):
        if cond["reference"]["select"] != "all":
            raise SystemExit(f"the reference of {name} is not the one of these conditions")
    cm.check_generator(wc["generator"])  # U6
    if c1["generator"] != wc["generator"]:
        raise SystemExit("U6: works 1 and 2 name different generators")
    for name, rec in (("words.json", words), ("work2_sp_color.json", main_rec), ("work1.json", w1)):
        if rec["meta"]["atact_commit"] != cm.installed_commit():
            raise SystemExit(f"U6: {name} was made with another generator")
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of work 2")
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if lex.sha256 != w2["lexicon"]["sha256"]:
        raise SystemExit("the lexicon differs from that of work 2")
    nrc = load_lexicon(ROOT / c1["lexicon"]["file"])
    if nrc.sha256 != c1["lexicon"]["sha256"] or w1["meta"]["lexicon_sha256"] != nrc.sha256:
        raise SystemExit("the NRC lexicon differs from that of work 1")
    vmap = calibration_map(cal_raw)
    rule = {"f": (mapping.F0, mapping.F0 + mapping.F1), "s": (mapping.S0, mapping.S0 + mapping.S1)}  # U6
    if not all(np.allclose(rule[q], want, rtol=0.0, atol=1e-12) for q, want in (("f", (3.0, 7.0)), ("s", (0.06, 0.30)))):
        raise SystemExit(f"U6: the range of the rule {rule} is not the one of the conditions")
    width = {q: hi - lo for q, (lo, hi) in rule.items()}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # U1: the constants ---------------------------------------------------
    k = CONSTANTS
    authors = list(words["authors"])
    if (len(authors) != k["authors"] or list(main_rec["tiles"]) != authors or len(w2["works"]) != k["works"]
            or {w["author"] for w in w2["works"]} != set(authors)):
        raise SystemExit("U1: the authors or works are not the constants")
    rows = {a: words["authors"][a]["words"] for a in authors}
    for a in authors:
        if (len(rows[a]) != k["words_per_author"]
                or [r["word"] for r in rows[a]] != main_rec["tiles"][a]):
            raise SystemExit(f"U1: {a}: the words of words.json are not those of the tiles of the main images")
    all_words = [r["word"] for a in authors for r in rows[a]]
    if (len(all_words) != k["tiles"] or len(set(all_words)) != k["distinct_words"]
            or main_rec["distinct_words"] != k["distinct_words"]):
        raise SystemExit("U1: the tiles or distinct words are not the constants")
    sents = w1["sentences"]
    if len(sents) != k["work1_sentences"] or w1["unmatched"] != 0 or any(p["unmatched"] for p in sents):
        raise SystemExit("U1: work 1 has not 104 sentences all matched")

    # U2: the words under map B -------------------------------------------
    u2 = 0
    for a in authors:
        for r in rows[a]:
            if r["word"] not in lex.entries:
                raise SystemExit(f"U2: {r['word']} is not an entry of the lexicon")
            V, A = (float(x) for x in vmap(*lex.entries[r["word"]]))
            if (r["V"], r["A"]) != (V, A):
                raise SystemExit(f"U2: {a} {r['word']}: V and A differ from map B")
            if tuple(float(x) for x in mapping.params(V, A)[:2]) != (r["f"], r["s"]):
                raise SystemExit(f"U2: {a} {r['word']}: f or s differ from params")
            u2 += 1

    # U3: the reference of the 23 works and the percentiles of the words ----
    done = {w["work_id"]: w for ws in json.loads((ROOT / "results/work2/work2.json").read_text())["authors"].values()
            for w in ws}
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
            raise SystemExit(f"U3: {w['work_id']}: the text step gives another result than results/work2")
        everything = japanese_text.text_path(t.sentences, lex, tagger, vmap, select=mc["reference"]["select"], order="word")
        counts.setdefault(w["author"], Counter()).update(x for p in everything for x in p.matched)
    values_rec = json.loads(path["values"].read_bytes())
    if ({a: sum(n.values()) for a, n in counts.items()} != values_rec["reference_tokens"]
            or main_rec["reference_tokens"] != values_rec["reference_tokens"]):
        raise SystemExit("U1: the tokens per author differ from those of values.json and the main images")
    wts = reference_weights(counts)
    ref_words = sorted(wts)
    if (len(ref_words) != k["reference_words"] or values_rec["reference_words"] != k["reference_words"]
            or main_rec["reference_words"] != k["reference_words"]):
        raise SystemExit(f"U1: {len(ref_words)} reference words, not {k['reference_words']}")
    rv = np.array([lex.entries[w][0] for w in ref_words])
    ra = np.array([lex.entries[w][1] for w in ref_words])
    weight = np.array([wts[w] for w in ref_words])
    if abs(weight.sum() - k["authors"]) > 1e-9:  # each author weighs 1
        raise SystemExit(f"the weights of the reference sum to {weight.sum()}, not {k['authors']}")
    u3 = 0
    for key, v in values_rec["values"].items():
        e = v["entry"]
        got = (weighted_percentile(rv, weight, lex.entries[e][0]), weighted_percentile(ra, weight, lex.entries[e][1]))
        if got != (v["p_valence_author_equal"], v["p_arousal_author_equal"]):
            raise SystemExit(f"U3: the percentiles of {key} recomputed differ from values.json")
        u3 += 1
    if not u3 or u3 != len(values_rec["values"]):
        raise SystemExit(f"U3: {u3} keys of values.json checked")
    absent = [r["word"] for a in authors for r in rows[a] if r["word"] not in wts]
    if absent:  # as works/work2_sp_color.py: every word of a tile is a word of the reference
        raise SystemExit(f"U3: {absent} are not words of the reference")
    VA_P = {a: [(2 * weighted_percentile(rv, weight, lex.entries[r["word"]][0]) - 1,
                 2 * weighted_percentile(ra, weight, lex.entries[r["word"]][1]) - 1) for r in rows[a]] for a in authors}
    u3_words = 0
    for a in authors:
        for r, (V, A) in zip(rows[a], VA_P[a]):
            v = values_rec["values"].get(r["word"])
            if v is not None and v["entry"] == r["word"]:
                if (V, A) != (2 * v["p_valence_author_equal"] - 1, 2 * v["p_arousal_author_equal"] - 1):
                    raise SystemExit(f"U3: {a} {r['word']}: the percentiles differ from values.json")
                u3_words += 1
    if not u3_words:
        raise SystemExit("U3: no word of the main images was checked against values.json")

    # U4: the reconstruction against reference_range.json -------------------
    p_v = np.array([weighted_percentile(rv, weight, x) for x in rv])
    p_a = np.array([weighted_percentile(ra, weight, x) for x in ra])
    tok_P = shapes(2 * p_v - 1, 2 * p_a - 1)
    u4 = 0
    if any(rr["sets"]["tokens_P"][q]["n"] != k["reference_words"] for q in QUANTITIES):
        raise SystemExit("U4: tokens_P of reference_range.json is not counted on the reference words")
    for q in QUANTITIES:
        q1, q3 = (float(x) for x in np.quantile(tok_P[q], [0.25, 0.75], method="inverted_cdf", weights=weight))
        got = {"min": float(tok_P[q].min()), "q1": q1, "q3": q3, "max": float(tok_P[q].max())}
        want = {x: rr["sets"]["tokens_P"][q][x] for x in got}
        if got != want:
            raise SystemExit(f"U4: tokens_P {q} recomputed {got} differs from reference_range.json {want}")
        u4 += 1

    # U5: the sentences of work 1 -------------------------------------------
    u5 = 0
    for i, p in enumerate(sents):
        terms = p["matched"]
        if not terms or p["Z"] != len(terms) or any(t not in nrc.entries for t in terms):
            raise SystemExit(f"U5: sentence {i}: the matched terms are not entries of the lexicon")
        z = len(terms)
        V = sum(nrc.entries[t][0] for t in terms) / z
        A = sum(nrc.entries[t][1] for t in terms) / z
        if (p["V"], p["A"]) != (V, A):
            raise SystemExit(f"U5: sentence {i}: V and A differ from the mean of its matched terms")
        if tuple(float(x) for x in mapping.params(V, A)[:2]) != (p["f"], p["s"]):
            raise SystemExit(f"U5: sentence {i}: f or s differ from params")
        u5 += 1

    # the sets ----------------------------------------------------------------
    main_G0 = {a: shapes([r["V"] for r in rows[a]], [r["A"] for r in rows[a]]) for a in authors}
    for a in authors:  # as recorded
        if any(not np.array_equal(main_G0[a][q], np.array([r[q] for r in rows[a]], dtype=float)) for q in QUANTITIES):
            raise SystemExit(f"U2: {a}: f or s of the set differ from words.json")
    main_P = {a: shapes([v for v, _ in VA_P[a]], [x for _, x in VA_P[a]]) for a in authors}
    nrc_terms = sorted(nrc.entries)
    lexicon_1 = shapes([nrc.entries[t][0] for t in nrc_terms], [nrc.entries[t][1] for t in nrc_terms])
    sentences_1 = shapes([p["V"] for p in sents], [p["A"] for p in sents])

    def table(sh: dict[str, np.ndarray]) -> dict:
        return {q: spread(sh[q], width[q]) for q in QUANTITIES}

    def pooled(per: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {q: np.concatenate([per[a][q] for a in authors]) for q in QUANTITIES}

    sets = {"main_G0": {**{a: table(main_G0[a]) for a in authors}, "pooled": table(pooled(main_G0))},
            "main_P": {**{a: table(main_P[a]) for a in authors}, "pooled": table(pooled(main_P))},
            "work1_sentences": table(sentences_1),
            "work1_lexicon": table(lexicon_1),
            "reference_sets_of_work2_as_recorded": {"lexicon_B": rr["sets"]["lexicon_B"],
                                                    "tokens_P": rr["sets"]["tokens_P"],
                                                    "method": "as recorded in reference_range.json: lexicon_B unweighted "
                                                              "(linear), tokens_P weighted (inverted_cdf)"}}
    if sets["work1_lexicon"]["f"]["n"] != len(nrc.entries):
        raise SystemExit("the lexicon of work 1 was not counted whole")

    tmp = Path(tempfile.mkdtemp(prefix=".paper_used_range-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {key: cm.sha256(p) for key, p in path.items()},
                        "lexicon_sha256": lex.sha256, "nrc_sha256": nrc.sha256, "dicdir_sha256": tagger.dicdir_sha256,
                        "library_sha256": cm.library_sha256(), **cm.versions("numpy", "fugashi", "ipadic"),
                        **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"U1_constants": {**CONSTANTS, "reference_tokens": values_rec["reference_tokens"]},
                          "U2_words_of_map_B_checked": u2,
                          "U3_percentiles_of_values_json_recomputed": u3,
                          "U3_words_of_the_main_images_checked_against_values_json": u3_words,
                          "U4_quantities_of_tokens_P_equal_reference_range": u4,
                          "U5_sentences_of_work1_checked": u5,
                          "U6_generator_and_rule": True,
                          "nrc_terms": len(nrc.entries), "nrc_skipped": list(nrc.skipped)},
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
