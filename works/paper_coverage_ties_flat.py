"""Four counts asked for by round 22 of the outline: (A) the tiles of work 1 and of figure 4 whose height or shading
has a span of zero, (B) the share of the running words of the story of work 1 covered by terms of the lexicon, (C) the
share of the distinct candidate words of each author of work 2, and of those passing the filter of the selection, found
in MEmoLon, (D) the largest weight of a group of tied values in the reference of the percentiles of work 2.

Usage:  uv run python works/paper_coverage_ties_flat.py   (refuses to overwrite)
Conditions: works/conditions/paper_coverage_ties_flat.json (committed before this script was written)
Input:  the records, figures and conditions named in the conditions, the text of work 1 through its record, the 23
        works of work 2 and the lexicons of both works (no drawing is written)
Output: results/paper_coverage_ties_flat/coverage_ties_flat.json
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
from PIL import Image  # noqa: E402
from atact import mapping  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.lexicon import load_lexicon, match_terms, tokenize  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import relief  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_coverage_ties_flat.json"
OUT_DIR = ROOT / "results" / "paper_coverage_ties_flat"
REF_COND = ROOT / "works" / "conditions" / "paper_reference_range.json"  # the conditions of the records read
FD_COND = ROOT / "works" / "conditions" / "paper_filtered_draws.json"
SCRIPTS = {"work1": "works/work1.py", "drawing": "works/work1_drawing.py", "words": "works/work2_words.py",
           "reference_range": "works/paper_reference_range.py"}
CODE = ["works/paper_coverage_ties_flat.py", *SCRIPTS.values(), "checks/author_texture.py", "works/_common.py",
        "src/relief.py", "src/pipeline.py", "src/japanese_text.py", "src/english_text.py", "src/window.py",
        "src/stretch.py"]
RECORD = "coverage_ties_flat.json"
CONSTANTS = {"work1_sentences": 104, "work1_terms": 1227, "figure4_tiles": 12, "figure4_rows": 4, "authors": 6,
             "works": 23, "reference_words": 18301, "candidate_tokens": 89336, "matched_tokens": 88591}
WORK1_IMAGES = ("common", "minmax", "common_eps")  # the figures of work 1, in the order of work1.FIGURES
Z_TOL_MM = 1e-9


def imported(name: str, rel: str):
    """A committed script imported as a module, checked to be that file."""
    mod = importlib.import_module(name)
    if Path(mod.__file__).resolve() != (ROOT / rel).resolve():
        raise SystemExit(f"the module imported is {mod.__file__}, not {rel}")
    return mod


def made_on(rec: dict, cond: Path, what: str) -> None:
    m = rec["meta"]
    if m["conditions_sha256"] != cm.sha256(cond) or m.get("git_dirty") is not False:
        raise SystemExit(f"{what} was not made on its conditions on a clean tree")


def span(x: np.ndarray) -> float:
    return float(x.max()) - float(x.min())


def spans_summary(spans: list[float], sentences: list[int]) -> dict:
    i = int(np.argmin(spans))
    return {"tiles": len(spans), "zero": int(sum(s == 0 for s in spans)), "smallest": spans[i],
            "smallest_sentence": sentences[i], "largest": max(spans)}


def share(n: int, d: int) -> float:
    if d <= 0:
        raise SystemExit("a share over nothing")
    return n / d


# ---------------------------------------------------------------- A: flat tiles


def flat_tiles(c1: dict, w1: dict, cd: dict, dr: dict, w1m, dm, figs: dict[str, np.ndarray], dimg: np.ndarray) -> dict:
    r = c1["relief"]
    d = w1m.dimensions(c1)
    eps, n, seed = relief.eps_for(d), r["grid_intervals"], r["seed"]
    if eps != w1["eps"] or n != w1["grid_intervals"]:
        raise SystemExit("K2: eps or the grid of work 1 differ from the record")
    if [(name, field, scale) for name, field, scale in w1m.FIGURES] != [
            ("common", "H", "common"), ("minmax", "H", "minmax"), ("common_eps", "H_eps", "common")]:
        raise SystemExit("the figures of work1.py are not the three named in the conditions")
    lay, fig1 = w1["layout"], c1["figure"]
    px = lay["tile_px"]
    k = CONSTANTS
    out = {key: [] for key in ("H_figure", "H_eps_figure", "H_eps_print", *(f"image_{x}" for x in WORK1_IMAGES))}
    sents = list(range(len(w1["sentences"])))
    k2 = 0
    for i, p in enumerate(w1["sentences"]):
        if [float(x) for x in mapping.params(p["V"], p["A"])] != [p["f"], p["s"], p["gamma"]]:
            raise SystemExit(f"sentence {i}: params differ from the record")
        tf = relief.tile(p["V"], p["A"], lambda f: procedural_field_separable(
            f, px, seed, octaves=relief.OCTAVES, normalize=relief.NORMALIZE), eps)
        y, x = dm.tile_place(i, lay, fig1)
        for name, field, scale in w1m.FIGURES:
            crop = figs[name][y : y + px, x : x + px]
            if scale == "minmax" and span(getattr(tf, field)) == 0:  # the record was scaled, so it was not flat
                raise SystemExit(f"K2: sentence {i}: recomputed flat, but {name}.png was drawn on the min-max scale")
            if not np.array_equal(dm.u8(relief.grey(getattr(tf, field), scale, d)), crop):
                raise SystemExit(f"K2: sentence {i}: the tile recomputed differs from {name}.png")
            out[f"image_{name}"].append(span(crop.astype(int)))
            k2 += 1
        tp = relief.tile(p["V"], p["A"], lambda f: relief.closed_field(f, n, seed), eps)
        top = relief.top_surface(tp.H_eps, "common", d)
        rec = p["stl"]["common"]
        if abs(float(top.min()) - rec["z_min_mm"]) > Z_TOL_MM or abs(float(top.max()) - rec["z_max_mm"]) > Z_TOL_MM:
            raise SystemExit(f"K2: sentence {i}: the heights on the print grid differ from the STL of the record")
        k2 += 1
        out["H_figure"].append(span(tf.H))
        out["H_eps_figure"].append(span(tf.H_eps))
        out["H_eps_print"].append(span(tp.H_eps))
    work1 = {key: spans_summary(v, sents) for key, v in out.items()}
    work1["stl_common_z_mm"] = spans_summary([p["stl"]["common"]["z_max_mm"] - p["stl"]["common"]["z_min_mm"]
                                              for p in w1["sentences"]], sents)

    # figure 4
    f = cd["figure"]
    kk, el, az_b, az_c = cd["drawings"]["supersample"], cd["drawings"]["elevation_deg"], cd["drawings"]["azimuth_b_deg"], cd["drawings"]["azimuth_c_deg"]
    gap, rl, rg = f["gap_px"], f["row_label_px"], f["row_gap_px"]
    n4 = cd["tiles"]["n"]
    if n4 != k["figure4_tiles"] or len(f["rows"]) != k["figure4_rows"] or len(dr["per_tile"]) != n4:
        raise SystemExit("K1: figure 4 has not 12 tiles in 4 rows")
    rows = {name: [] for name in "abcd"}
    arrays = {name: [] for name in ("b", "c")}
    for j in range(n4):
        p = w1["sentences"][j]
        if dr["per_tile"][j]["sentence"] != j:
            raise SystemExit(f"K1: tile {j} of drawing.json is not sentence {j}")
        t = relief.tile(p["V"], p["A"], lambda f_: procedural_field_separable(
            f_, px * kk, seed, octaves=relief.OCTAVES, normalize=relief.NORMALIZE), eps)
        z = relief.scaled(t.H_eps, "common", d)
        b = dm.block_mean(dm.lambert(z, d.L_mm, az_b, el), kk)
        cc = dm.block_mean(dm.lambert(z, d.L_mm, az_c, el), kk)
        ta = relief.tile(p["V"], p["A"], lambda f_: procedural_field_separable(
            f_, px, seed, octaves=relief.OCTAVES, normalize=relief.NORMALIZE), eps)
        a = relief.grey(ta.H_eps, "common", d)
        for r_, (name, g) in enumerate(zip("abcd", (a, b, cc, None))):
            y0, x0 = r_ * (rl + px + rg) + rl, j * (px + gap)
            crop = dimg[y0 : y0 + px, x0 : x0 + px]
            if name == "d":
                if span(b) == 0:  # the record was stretched, so it was not flat
                    raise SystemExit(f"K2: tile {j}: recomputed flat, but drawing.png was stretched")
                g = dm.stretch(b)
            if not np.array_equal(dm.u8(g), crop):
                raise SystemExit(f"K2: tile {j}: drawing {name} recomputed differs from drawing.png")
            rows[name].append(span(crop.astype(int)))
            k2 += 1
        arrays["b"].append(span(b))
        arrays["c"].append(span(cc))
    if k2 != (len(WORK1_IMAGES) + 1) * k["work1_sentences"] + k["figure4_rows"] * n4:
        raise SystemExit(f"K2: {k2} comparisons, not one per image, STL and drawing")
    t4 = list(range(n4))
    figure4 = {**{f"image_row_{name}": spans_summary(v, t4) for name, v in rows.items()},
               **{f"shading_{name}": spans_summary(v, t4) for name, v in arrays.items()},
               "recorded_sd_zero": {name: int(sum(t[name]["sd"] == 0 for t in dr["per_tile"])) for name in "abcd"},
               "recorded_sd_smallest": {name: min(t[name]["sd"] for t in dr["per_tile"]) for name in "abcd"}}
    return {"work1": work1, "figure4": figure4, "K2_compared": k2}


# ---------------------------------------------------------------- B: work 1 coverage


def work1_coverage(w1: dict, lex) -> dict:
    k = CONSTANTS
    total = covered = multi = terms = 0
    per = []
    for i, p in enumerate(w1["sentences"]):
        toks = tokenize(p["text"])
        found = match_terms(toks, lex)
        if found != p["matched"] or len(found) != p["Z"]:
            raise SystemExit(f"K3: sentence {i}: the terms found again differ from the record")
        cov = sum(len(t.split(" ")) for t in found)
        if cov > len(toks):
            raise SystemExit(f"sentence {i}: more words covered than words")
        total, covered, terms = total + len(toks), covered + cov, terms + len(found)
        multi += sum(" " in t for t in found)
        per.append(share(cov, len(toks)))
    if terms != k["work1_terms"] or len(per) != k["work1_sentences"]:
        raise SystemExit("K1: the terms or sentences of work 1 are not the constants")
    dropped = w1["story"]["dropped_pieces"]
    q = np.quantile(per, [0.25, 0.5, 0.75])
    return {"running_tokens": total, "covered_tokens": covered, "share_covered": share(covered, total),
            "terms": terms, "terms_of_more_than_one_word": multi,
            "per_sentence_share": {"min": min(per), "q1": float(q[0]), "median": float(q[1]), "q3": float(q[2]),
                                   "max": max(per)},
            "dropped_pieces": len(dropped), "tokens_in_dropped_pieces": sum(len(tokenize(x)) for x in dropped)}


# ---------------------------------------------------------------- main


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    if c["constants"] != CONSTANTS:
        raise SystemExit("the constants of the conditions differ from those of the script")
    if not c["outputs"]["record"].startswith(RECORD + ":"):
        raise SystemExit("the record of the conditions differs from that of the script")
    inp = c["inputs"]
    if inp["scripts"] != SCRIPTS or inp["work1_figures"]["files"] != [f"{x}.png" for x in WORK1_IMAGES]:
        raise SystemExit("the scripts or figures named in the conditions are not those of this script")
    path = {key: ROOT / v["path"] for key, v in inp.items() if isinstance(v, dict) and "path" in v}
    fig_paths = {x: path["work1_figures"] / f"{x}.png" for x in WORK1_IMAGES}
    files = [p for key, p in path.items() if key != "work1_figures"] + list(fig_paths.values())
    cm.check_file(path["values"], inp["values"]["sha256"], "values")
    c1 = json.loads(path["work1_conditions"].read_bytes())
    w2 = json.loads(path["work2_conditions"].read_bytes())
    cm.same_as_head([COND, REF_COND, FD_COND, *files, ROOT / w2["map"]["file"], *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work1", "results/work1_drawing", "results/work2",
                          "results/work2_words", "results/sketch_sp_color", "results/paper_reference_range",
                          "results/paper_filtered_draws", "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    w1m = imported("work1", SCRIPTS["work1"])
    dm = imported("work1_drawing", SCRIPTS["drawing"])
    ww = imported("work2_words", SCRIPTS["words"])
    rr = imported("paper_reference_range", SCRIPTS["reference_range"])
    for mod, rel in ((relief, "src/relief.py"), (japanese_text, "src/japanese_text.py"), (cm, "works/_common.py")):
        if Path(mod.__file__).resolve() != (ROOT / rel).resolve():
            raise SystemExit(f"the module imported is {mod.__file__}, not {rel}")
    for name, rel in (("pipeline", "src/pipeline.py"), ("window", "src/window.py"), ("stretch", "src/stretch.py"),
                      ("english_text", "src/english_text.py"), ("author_texture", "checks/author_texture.py")):
        if Path(sys.modules[name].__file__).resolve() != (ROOT / rel).resolve():
            raise SystemExit(f"the module {name} imported is not {rel}")
    cm.check_generator(c1["generator"])
    k = CONSTANTS

    # the records read, each made on its conditions on a clean tree
    w1 = json.loads(path["work1_result"].read_bytes())
    made_on(w1, path["work1_conditions"], "work1.json")
    if any(w1["figures"][x]["sha256"] != cm.sha256(p) for x, p in fig_paths.items()):
        raise SystemExit("a figure of work 1 differs from its record")
    cd = json.loads(path["drawing_conditions"].read_bytes())
    dr = json.loads(path["drawing_result"].read_bytes())
    made_on(dr, path["drawing_conditions"], "drawing.json")
    if dr["figures"] != {cd["figure"]["file"]: cm.sha256(path["drawing_figure"])} or path["drawing_figure"].name != cd["figure"]["file"]:
        raise SystemExit("drawing.png differs from its record")
    if cd["inputs"]["work1_sha256"] != cm.sha256(path["work1_result"]):
        raise SystemExit("figure 4 was not drawn from this work1.json")
    w2r = json.loads(path["work2_result"].read_bytes())
    made_on(w2r, path["work2_conditions"], "work2.json")
    wc = json.loads(path["words_conditions"].read_bytes())
    words_rec = json.loads(path["words"].read_bytes())
    made_on(words_rec, path["words_conditions"], "words.json")
    if wc["work2_conditions_sha256"] != cm.sha256(path["work2_conditions"]):
        raise SystemExit("work2.json is not the one the words were selected on")
    rc = json.loads(path["reference_conditions"].read_bytes())
    if rc["inputs"]["work2_conditions"] != "works/conditions/work2.json" or rc["reference"]["select"] != "all":
        raise SystemExit("the reference of sketch_sp_color is not the one of these conditions")
    ref_rec = json.loads(path["reference_range"].read_bytes())
    fd = json.loads(path["filtered_draws"].read_bytes())
    for rec, cond, what in ((ref_rec, REF_COND, "reference_range.json"), (fd, FD_COND, "filtered_draws.json")):
        made_on(rec, cond, what)
        if any(rec["meta"]["inputs_sha256"][key] != cm.sha256(path[key])
               for key in ("values", "work2_conditions", "work2_result", "reference_conditions", "calibration")):
            raise SystemExit(f"{what} was not made from the inputs named here as committed")
    if fd["meta"]["inputs_sha256"]["words"] != cm.sha256(path["words"]):
        raise SystemExit("filtered_draws.json was not made from this words.json")
    if ref_rec["meta"]["script_sha256"] != cm.sha256(ROOT / SCRIPTS["reference_range"]):
        raise SystemExit("reference_range.json was not made by the script imported here")
    if cm.versions("numpy")["numpy"] != ref_rec["meta"]["numpy"]:
        raise SystemExit("numpy is not the version of reference_range.json")
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    jlex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if (tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"] or jlex.sha256 != w2["lexicon"]["sha256"]
            or any(m["lexicon_sha256"] != jlex.sha256 for m in (w2r["meta"], words_rec["meta"], ref_rec["meta"]))):
        raise SystemExit("the Japanese lexicon or the MeCab dictionary differs from that of work 2 or of a record read")
    elex = load_lexicon(ROOT / c1["lexicon"]["file"])
    if elex.sha256 != c1["lexicon"]["sha256"] or w1["meta"]["lexicon_sha256"] != elex.sha256:
        raise SystemExit("the English lexicon differs from that of work 1")
    cal_raw = cm.check_file(path["calibration"], w2["map"]["sha256"], "the calibration (the map of work 2)")
    if json.loads(cal_raw)["meta"].get("git_dirty") is not False:
        raise SystemExit("calibrate_memolon.json was not computed on a clean tree")
    vmap = calibration_map(cal_raw)
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if len(w1["sentences"]) != k["work1_sentences"]:
        raise SystemExit("K1: work 1 has not 104 sentences")

    # A and B --------------------------------------------------------------
    figs = {x: np.asarray(Image.open(p).convert("L")) for x, p in fig_paths.items()}
    dimg = np.asarray(Image.open(path["drawing_figure"]).convert("L"))
    A = flat_tiles(c1, w1, cd, dr, w1m, dm, figs, dimg)
    B = work1_coverage(w1, elex)

    # C and D: the text step of work 2, per work ----------------------------
    done = {w["work_id"]: w for ws in w2r["authors"].values() for w in ws}
    cand: dict[str, Counter] = {}  # per work, the candidate forms (matched or not)
    matched_words, author_of, tokens = {}, {}, {}
    counts: dict[str, Counter] = {}  # per author, the reference (selection 'all')
    k4 = 0
    if len(w2["works"]) != k["works"]:
        raise SystemExit("K1: not 23 works")
    for w in w2["works"]:
        cm.check_file(ROOT / w["zip"], w["zip_sha256"], f"the zip of {w['work_id']}")
        name, data = read_aozora_zip(ROOT / w["zip"])
        if name != w["txt_name"]:
            raise SystemExit(f"{w['work_id']}: the zip holds {name}")
        text = data.decode(w["encoding"])
        body, _, _, kept, _ = work2_body(text)
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != w["body_sha256"] or len(kept) != w["sentences"]:
            raise SystemExit(f"{w['work_id']}: the body differs from work 2")
        t = work2_text(text, jlex, tagger, vmap)
        ref = done[w["work_id"]]
        forms = Counter(x.surface for s in t.sentences for x in tagger.tokens(s) if japanese_text.is_candidate(x, "pos"))
        n_matched = sum(n for x, n in forms.items() if x in jlex.entries)
        if (len(t.points) != ref["sentences"] or sum(p.Z for p in t.points) != ref["matched_tokens"]
                or sum(forms.values()) != ref["candidate_tokens"] or n_matched != ref["matched_tokens"]
                or sum(p.n_candidates for p in t.points) != ref["candidate_tokens"]):
            raise SystemExit(f"K4: {w['work_id']}: the candidates or matched tokens differ from work2.json")
        k4 += 1
        cand[w["work_id"]] = forms
        matched_words[w["work_id"]] = Counter(x for p in t.points for x in p.matched)
        if matched_words[w["work_id"]] != Counter({x: n for x, n in forms.items() if x in jlex.entries}):
            raise SystemExit(f"K4: {w['work_id']}: the matched words are not the candidate forms in MEmoLon")
        author_of[w["work_id"]] = w["author"]
        tokens[w["work_id"]] = sum(matched_words[w["work_id"]].values())  # the tokens of select
        everything = japanese_text.text_path(t.sentences, jlex, tagger, vmap, select=rc["reference"]["select"], order="word")
        counts.setdefault(w["author"], Counter()).update(x for p in everything for x in p.matched)
    if (sum(x["candidate_tokens"] for x in done.values()), sum(x["matched_tokens"] for x in done.values())) != (
            k["candidate_tokens"], k["matched_tokens"]):
        raise SystemExit("K1: the candidate and matched tokens of work2.json are not the constants")

    # C
    authors = list(words_rec["authors"])
    if len(authors) != k["authors"] or set(authors) != set(author_of.values()):
        raise SystemExit("K1: the authors are not the six of words.json")
    sel = wc["selection"]
    chosen = ww.select(matched_words, author_of, tokens, sel["min_count"], sel["min_works"], sel["max_share"], sel["n_words"])
    if any(chosen[a]["candidates"] != words_rec["authors"][a]["candidates"] for a in authors):
        raise SystemExit("K4: the selection again does not give the candidates of words.json")
    if not (sel["min_count"] >= 1 and sel["min_works"] >= 1):  # else a form absent from the author could pass
        raise SystemExit("the filter of the selection admits a word absent from the author")

    def unsigned(words_per_work: dict[str, Counter]) -> dict[str, set[str]]:
        """The forms passing the filter of select itself, with every z set to 1 so that the sign passes every form."""
        real = ww.log_odds
        try:
            ww.log_odds = lambda o, r, pr: {w: 1.0 for w in pr}
            got = ww.select(words_per_work, author_of, tokens, sel["min_count"], sel["min_works"], sel["max_share"],
                            10**9)
        finally:
            ww.log_odds = real
        return {a: {d["word"] for d in got[a]["words"]} for a in authors}

    unsigned_cand = unsigned(cand)  # the rate of a work is its count over the matched tokens of the work (tokens)
    unsigned_matched = unsigned(matched_words)  # the filtered pool of filtered_draws.json
    C: dict = {"authors": {}}
    union_forms: set[str] = set()
    for a in authors:
        own = [w for w in cand if author_of[w] == a]
        forms_a = sum((cand[w] for w in own), Counter())
        union_forms |= set(forms_a)
        in_lex = [x for x in forms_a if x in jlex.entries]
        passing = unsigned_cand[a]
        if not passing <= set(forms_a):
            raise SystemExit(f"K4: {a}: a form passing the filter is not a candidate of the author")
        passing_in = {x for x in passing if x in jlex.entries}
        if passing_in != unsigned_matched[a] or len(passing_in) != fd["sets"]["pool_sizes"]["filtered"][a]:
            raise SystemExit(f"K4: {a}: the forms in MEmoLon passing the filter are not the filtered pool")
        cand_tok = sum(forms_a.values())
        match_tok = sum(n for x, n in forms_a.items() if x in jlex.entries)
        C["authors"][a] = {"distinct": len(forms_a), "distinct_in_lexicon": len(in_lex),
                           "share_distinct": share(len(in_lex), len(forms_a)),
                           "passing_filter": len(passing), "passing_filter_in_lexicon": len(passing_in),
                           "share_passing_filter": share(len(passing_in), len(passing)),
                           "candidate_tokens": cand_tok, "matched_tokens": match_tok,
                           "share_running": share(match_tok, cand_tok)}
    u_in = sum(x in jlex.entries for x in union_forms)
    C["six_authors_together"] = {"distinct": len(union_forms), "distinct_in_lexicon": u_in,
                                 "share_distinct": share(u_in, len(union_forms))}

    # D
    vrec = json.loads(path["values"].read_bytes())
    if len(counts) != k["authors"] or {a: sum(n.values()) for a, n in counts.items()} != vrec["reference_tokens"]:
        raise SystemExit("K5: the tokens per author differ from those of values.json")
    wts = rr.reference_weights(counts)
    ref_words = sorted(wts)
    if len(ref_words) != k["reference_words"] or vrec["reference_words"] != k["reference_words"]:
        raise SystemExit(f"K5: {len(ref_words)} reference words")
    weight = np.array([wts[w]["author_equal"] for w in ref_words])
    total_w = float(weight.sum())
    if abs(total_w - k["authors"]) > 1e-9:
        raise SystemExit(f"K5: the weights sum to {total_w}")
    D: dict = {"total_weight": total_w}
    p_dims = {}
    for i, dim in enumerate(("valence", "arousal")):
        vals = np.array([jlex.entries[w][i] for w in ref_words])
        groups: dict[float, list[int]] = {}
        for j, v in enumerate(vals.tolist()):
            groups.setdefault(v, []).append(j)
        gw = {v: float(weight[ix].sum()) for v, ix in groups.items()}
        top = max(gw, key=lambda v: (gw[v], -v))
        multi = [v for v, ix in groups.items() if len(ix) > 1]
        D[dim] = {"distinct_values": len(groups), "groups_of_more_than_one_word": len(multi),
                  "largest_group_share": gw[top] / total_w, "largest_group_words": len(groups[top]),
                  "largest_single_word_share": float(weight.max()) / total_w,
                  "share_in_groups_of_more_than_one_word": sum(gw[v] for v in multi) / total_w}
        p_dims[dim] = np.array([rr.weighted_percentile(vals, weight, x) for x in vals])
        # the groups tie as weighted_percentile ties: below plus half the group, over the total, for every group
        order = sorted(gw)
        below = np.concatenate([[0.0], np.cumsum([gw[v] for v in order])[:-1]])
        for v, b in zip(order, below, strict=True):
            if not np.isclose((b + 0.5 * gw[v]) / total_w, p_dims[dim][groups[v][0]], rtol=0.0, atol=1e-12):
                raise SystemExit(f"D: {dim}: a group of equal values is not a tie of weighted_percentile")
    sh = rr.shapes(2 * p_dims["valence"] - 1, 2 * p_dims["arousal"] - 1)
    rule = {"f": (mapping.F0, mapping.F0 + mapping.F1), "s": (mapping.S0, mapping.S0 + mapping.S1)}
    if ref_rec["rule_range"] != {q: list(v) for q, v in rule.items()}:
        raise SystemExit("the range of the rule of reference_range.json is not the installed one")
    keep = weight > 0
    again = {q: rr.spread(sh[q][keep], rule[q][1] - rule[q][0], weight[keep]) for q in ("f", "s")}
    if again != ref_rec["sets"]["tokens_P"]:
        raise SystemExit("K5: tokens_P recomputed differs from reference_range.json")
    D["tokens_P_recomputed_equals_record"] = True

    tmp = Path(tempfile.mkdtemp(prefix=".paper_coverage_ties_flat-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {key: cm.sha256(p) for key, p in path.items() if p.is_file()}
                        | {f"work1_figure_{x}": cm.sha256(p) for x, p in fig_paths.items()},
                        "code_sha256": {rel: cm.sha256(ROOT / rel) for rel in CODE},
                        "conditions_of_records_read_sha256": {str(p.relative_to(ROOT)): cm.sha256(p)
                                                              for p in (REF_COND, FD_COND)},
                        "english_lexicon_sha256": elex.sha256, "japanese_lexicon_sha256": jlex.sha256,
                        "dicdir_sha256": tagger.dicdir_sha256, "library_sha256": cm.library_sha256(),
                        **cm.versions("numpy", "pillow", "fugashi", "ipadic"), **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"K1_constants": CONSTANTS, "K2_tiles_compared_with_the_records": A.pop("K2_compared"),
                          "K3_sentences_of_work1_matched_again": len(w1["sentences"]),
                          "K4_works_counted_again": k4, "K5_reference_and_tokens_P_again": True,
                          "D_groups_tie_as_weighted_percentile_added_after_review": True},
               "A_flat_tiles": A, "B_work1_coverage": B, "C_work2_coverage": C, "D_ties_of_the_percentiles": D}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        with (tmp / RECORD).open("x") as fh:
            fh.write(text)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({k_: res[k_] for k_ in ("checks", "B_work1_coverage")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
