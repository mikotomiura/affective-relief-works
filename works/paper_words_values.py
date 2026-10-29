"""The f and s of each of the 120 words of the main images of work 2 under map B and the author-equal percentiles,
kept word by word, so that the script of figure 8 can compute again the middle half it shows. The reconstruction of the
reference and its checks U1 to U4 are copied from works/paper_used_range.py (lines 88 to 256 of its main, unchanged
but for the name of its conditions); its functions are imported from it.

Usage:  uv run python works/paper_words_values.py   (refuses to overwrite)
Conditions: works/conditions/paper_words_values.json (committed before this script was written)
Input:  the inputs of works/conditions/paper_used_range.json and results/paper_used_range/used_range.json
Output: results/paper_words_values/
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
from paper_used_range import (CODE, CONSTANTS, QUANTITIES, RECORD, reference_weights, shapes, spread,  # noqa: E402
                              weighted_percentile)
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_words_values.json"
UR_COND = ROOT / "works" / "conditions" / "paper_used_range.json"
UR_SCRIPT = ROOT / "works" / "paper_used_range.py"
OUT_DIR = ROOT / "results" / "paper_words_values"
OWN_CODE = ["works/paper_words_values.py"]
OUT_RECORD = "words_values.json"
U_COUNTS = {"U2": 120, "U3_keys": 64, "U3_words": 44, "U4": 2}


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    own_raw, own = cm.load_conditions(COND)
    if not own["outputs"]["record"].startswith(OUT_RECORD + ":"):
        raise SystemExit("the record of the conditions differs from that of the script")
    own_path = {k: ROOT / v["path"] for k, v in own["inputs"].items() if isinstance(v, dict)}
    if (own_path["used_range_conditions"] != UR_COND or own_path["used_range_script"] != UR_SCRIPT
            or own_path["used_range"] != ROOT / "results" / "paper_used_range" / "used_range.json"):
        raise SystemExit("the inputs of the conditions are not those of the script")
    cm.same_as_head([COND, *own_path.values(), *(ROOT / p for p in OWN_CODE)])
    ur = json.loads(own_path["used_range"].read_bytes())
    if (ur["meta"]["conditions_sha256"] != cm.sha256(UR_COND) or ur["meta"]["script_sha256"] != cm.sha256(UR_SCRIPT)
            or ur["meta"].get("git_dirty") is not False):
        raise SystemExit("used_range.json was not made by these conditions and this script on a clean tree")
    if ur["meta"]["atact_commit"] != cm.installed_commit():
        raise SystemExit("used_range.json was made with another generator")

    # copied from works/paper_used_range.py, lines 88 to 256 ------------------
    raw, c = cm.load_conditions(UR_COND)
    if c["constants"] != CONSTANTS:
        raise SystemExit("the constants of the conditions differ from those of the script")
    if tuple(c["statistics"]["quantities"]) != QUANTITIES or not c["outputs"]["record"].startswith(RECORD + ":"):
        raise SystemExit("the quantities or the record of the conditions differ from those of the script")
    inp = c["inputs"]
    path = {k: ROOT / v["path"] for k, v in inp.items() if isinstance(v, dict)}
    cm.check_file(path["values"], inp["values"]["sha256"], "values")
    w2 = json.loads(path["work2_conditions"].read_bytes())
    c1 = json.loads(path["work1_conditions"].read_bytes())
    cm.same_as_head([UR_COND, *path.values(), ROOT / w2["map"]["file"], ROOT / "results/work2/work2.json",
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


    # end of the copy ------------------------------------------------------------
    # the record of used_range was made from these inputs (beyond its conditions and script)
    made_from = {"inputs_sha256": {key: cm.sha256(q) for key, q in path.items()}, "library_sha256": cm.library_sha256(),
                 "lexicon_sha256": lex.sha256, "nrc_sha256": nrc.sha256, "dicdir_sha256": tagger.dicdir_sha256}
    if any(ur["meta"].get(key) != v for key, v in made_from.items()):
        raise SystemExit(f"used_range.json was not made from these inputs: "
                         f"{[key for key, v in made_from.items() if ur['meta'].get(key) != v]}")

    # the checks of the copy against the record ------------------------------
    got = {"U2": u2, "U3_keys": u3, "U3_words": u3_words, "U4": u4}
    rec = ur["checks"]
    want = {"U2": rec["U2_words_of_map_B_checked"], "U3_keys": rec["U3_percentiles_of_values_json_recomputed"],
            "U3_words": rec["U3_words_of_the_main_images_checked_against_values_json"],
            "U4": rec["U4_quantities_of_tokens_P_equal_reference_range"]}
    if got != want or got != U_COUNTS:  # V3
        raise SystemExit(f"V3: the counts of U1 to U4 {got} differ from the record {want} or from {U_COUNTS}")

    # the rows (V1) -------------------------------------------------------------
    out_rows = []
    src_of = {}
    for a in authors:
        if len({r["word"] for r in rows[a]}) != len(rows[a]):
            raise SystemExit(f"V1: {a} has a word twice")
        for r, (V, A) in zip(rows[a], VA_P[a], strict=True):
            src_of[(a, r["word"])] = r
            pv = weighted_percentile(rv, weight, lex.entries[r["word"]][0])
            pa = weighted_percentile(ra, weight, lex.entries[r["word"]][1])
            if (2 * pv - 1, 2 * pa - 1) != (V, A):
                raise SystemExit(f"V1: {a} {r['word']}: the percentiles differ from those of the copy")
            fs = shapes([V], [A])
            out_rows.append({"author": a, "word": r["word"],
                             "map_B": {"V": r["V"], "A": r["A"], "f": r["f"], "s": r["s"]},
                             "percentiles": {"p_valence": pv, "p_arousal": pa, "V": V, "A": A,
                                             "f": float(fs["f"][0]), "s": float(fs["s"][0])}})
    if (len(out_rows) != k["tiles"] or [x["word"] for x in out_rows] != [w for a in authors for w in main_rec["tiles"][a]]
            or len({x["author"] for x in out_rows}) != k["authors"]):
        raise SystemExit("V1: the rows are not 120 in the order of the tiles")

    for x in out_rows:  # each row as recorded and as params gives it
        src = src_of[(x["author"], x["word"])]
        if x["map_B"] != {q: src[q] for q in ("V", "A", "f", "s")}:
            raise SystemExit(f"V1: {x['author']} {x['word']}: map B differs from words.json")
        for version in ("map_B", "percentiles"):
            v = x[version]
            if tuple(float(y) for y in mapping.params(v["V"], v["A"])[:2]) != (v["f"], v["s"]):
                raise SystemExit(f"V1: {x['author']} {x['word']}: f or s of {version} differ from params")

    # V2: the shares from the rows equal used_range.json --------------------------
    v2 = 0
    for version, set_name in (("map_B", "main_G0"), ("percentiles", "main_P")):
        per = {a: {q: np.array([x[version][q] for x in out_rows if x["author"] == a], dtype=float) for q in QUANTITIES}
               for a in authors}
        per["pooled"] = {q: np.array([x[version][q] for x in out_rows], dtype=float) for q in QUANTITIES}
        for who, sh in per.items():
            for q in QUANTITIES:
                if spread(sh[q], width[q]) != ur["sets"][set_name][who][q]:
                    raise SystemExit(f"V2: {set_name} {who} {q} from the rows differs from used_range.json")
                v2 += 1

    tmp = Path(tempfile.mkdtemp(prefix=".paper_words_values-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(own_raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "used_range_conditions_sha256": cm.sha256(UR_COND), "used_range_script_sha256": cm.sha256(UR_SCRIPT),
                        "inputs_sha256": {key: cm.sha256(p) for key, p in {**path, **own_path}.items()},
                        "work2_record_sha256": cm.sha256(ROOT / "results/work2/work2.json"),
                        "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                        "lexicon_sha256": lex.sha256, "nrc_sha256": nrc.sha256, "dicdir_sha256": tagger.dicdir_sha256,
                        "zips": "checked against the SHA-256 of works/conditions/work2.json (inputs_sha256.work2_conditions)",
                        "atact_commit": cm.installed_commit(), "library_sha256": cm.library_sha256(),
                        **cm.versions("numpy", "fugashi", "ipadic"), **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"V1_rows": len(out_rows), "V2_quantities_equal_used_range": v2, "V3_counts_of_U1_to_U4": got,
                          "used_range_made_from_these_inputs": True,
                          "what_V2_covers": "V2 compares n, the minimum, the quartiles (linear) and the maximum of every "
                                            "set, so it checks the values at those positions only; the other rows rest on "
                                            "the copy of the reconstruction being the committed one and its inputs being "
                                            "those of used_range.json (checked above), and U3 checks 44 of the words "
                                            "against values.json"},
               "rows": out_rows}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        with (tmp / OUT_RECORD).open("x", encoding="utf-8") as fh:
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
