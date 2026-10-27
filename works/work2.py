"""Work 2: for each chosen Japanese author, one trajectory per work from the 40-sentence window means, unstretched
and under the common stretch. The portrait is not drawn here (its form is still to be decided).

Usage:  uv run python works/work2.py        (refuses to overwrite; stops if an input is still "FILL")
Conditions: works/conditions/work2.json (the template works/conditions/work2.template.json with the chosen works
        filled in, committed before this script is run on them)
Input:  the Aozora Bunko zips, the Japanese MEmoLon lexicon (external/, not in the repository),
        results/calibrate_memolon.json (map B) and results/reference_stretch.json (the factor k)
Output: results/work2/work2.json

Steps: src/pipeline.work2_text (strip_aozora -> strip_date_note -> split_sentences -> sentence_point with select
"pos", order "word" and map B -> trajectory, W = 40), then src/pipeline.work2_trajectories (stretch by k).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import window  # noqa: E402
from pipeline import Draw, calibration_map, read_aozora_zip, work2_body, work2_text, work2_trajectories  # noqa: E402

COND = ROOT / "works" / "conditions" / "work2.json"
TEMPLATE = ROOT / "works" / "conditions" / "work2.template.json"
OUT_DIR = ROOT / "results" / "work2"
CODE = ["works/work2.py", "works/_common.py", "src/pipeline.py", "src/english_text.py", "src/japanese_text.py",
        "src/window.py", "src/stretch.py", "src/relief.py"]


def validate(c: dict) -> None:
    works = c["works"]
    if not isinstance(works, list) or not works:
        raise SystemExit("works must be a non-empty list")
    lo, hi = cm.need(c["range"], "min_sentences", int), cm.need(c["range"], "max_sentences", int)
    ids = set()
    for i, w in enumerate(works):
        where = f"works[{i}]."
        for k in ("author", "work_id", "title", "zip", "zip_sha256", "txt_name", "encoding", "body_sha256"):
            cm.need(w, k, str, where)
        if w["encoding"] != "cp932":
            raise SystemExit(f"{where}encoding: the Aozora Bunko text files are read as cp932")
        if w["work_id"] in ids:
            raise SystemExit(f"work {w['work_id']} is listed twice")
        ids.add(w["work_id"])
        if not lo <= cm.need(w, "sentences", int, where) <= hi:
            raise SystemExit(f"{where}sentences outside the range {lo}-{hi}")
        if w.get("date_note") is not None:
            cm.need(w, "date_note", str, where)
    authors = {w["author"] for w in works}
    if not cm.need(c["range"], "min_authors", int) <= len(authors) <= cm.need(c["range"], "max_authors", int):
        raise SystemExit(f"{len(authors)} authors, outside the range of the conditions")
    for a in authors:
        if sum(w["author"] == a for w in works) < cm.need(c["range"], "min_works_per_author", int):
            raise SystemExit(f"fewer works than the minimum for {a}")
    wd = c["window"]
    if cm.need(wd, "W", int, "window.") != window.W_WORK2 or cm.need(wd, "min_matched_fraction", float, "window.") \
            != window.MIN_MATCHED_FRACTION:
        raise SystemExit("the window differs from that of src/window.py")
    k = cm.need(c["stretch"], "k", float, "stretch.")
    if not (k > 0 and math.isfinite(k)):
        raise SystemExit(f"stretch factor {k}")
    st = c["step"]
    if st.get("select") != "pos" or st.get("order") != "word":
        raise SystemExit("the step is select 'pos', order 'word'")


def run(c: dict, root: Path, lex: japanese_text.Lexicon, tagger: japanese_text.Tagger,
        vmap: japanese_text.ValueMap, draw: Draw | None = None, out_dir: Path | None = None) -> dict:
    validate(c)
    k = c["stretch"]["k"]
    authors: dict[str, list[dict]] = {}
    for w in c["works"]:
        cm.check_file(root / w["zip"], w["zip_sha256"], f"the zip of {w['work_id']}")
        name, raw = read_aozora_zip(root / w["zip"])
        if name != w["txt_name"]:
            raise SystemExit(f"{w['work_id']}: the zip holds {name}, not {w['txt_name']}")
        text = raw.decode("cp932")
        # the text step without the lexicon is checked first
        body, _, note, kept, _ = work2_body(text)
        if (hashlib.sha256(body.encode("utf-8")).hexdigest() != w["body_sha256"] or len(kept) != w["sentences"]
                or note != w.get("date_note")):
            raise SystemExit(f"{w['work_id']}: the body, the number of sentences or the date note differs from the "
                             "conditions")
        t = work2_text(text, lex, tagger, vmap)
        matched = sum(not p.unmatched for p in t.points)
        rec = {"work_id": w["work_id"], "title": w["title"], "sentences": len(t.sentences),
               "matched_sentences": matched, "matched_share": matched / len(t.points),
               "candidate_tokens": sum(p.n_candidates for p in t.points), "matched_tokens": sum(p.Z for p in t.points),
               "dropped_pieces": t.dropped, "date_note": t.date_note, **work2_trajectories(t.trajectory, k)}
        authors.setdefault(w["author"], []).append(rec)
    res = {"k": k, "authors": authors}
    if draw is not None and out_dir is not None:
        res["drawn"] = {a: [p.name for p in draw({"author": a, "works": ws}, out_dir)] for a, ws in authors.items()}
    return res


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    validate(c)
    st = c["stretch"]
    cm.same_as_head([COND, TEMPLATE, ROOT / c["map"]["file"], ROOT / st["file"], *(ROOT / p for p in CODE)])
    cm.check_template(TEMPLATE, c)
    state = cm.git_state(["works", "src", "results/calibrate_memolon.json", "results/reference_stretch.json",
                          "pyproject.toml", "uv.lock", ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_generator(c["generator"])
    vmap = calibration_map(cm.check_file(ROOT / c["map"]["file"], c["map"]["sha256"], "the calibration"))
    ref = json.loads(cm.check_file(ROOT / st["file"], st["sha256"], "the reference result"))
    if ref["result"]["k"] != st["k"]:
        raise SystemExit("k differs from the value of the reference result")
    cm.check_versions(c["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != c["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of the conditions")
    lex = japanese_text.load_memolon(ROOT / c["lexicon"]["file"])
    if lex.sha256 != c["lexicon"]["sha256"]:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    res = run(c, ROOT, lex, tagger, vmap)
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "lexicon_sha256": lex.sha256,
                    "dicdir_sha256": tagger.dicdir_sha256, "uv_lock_sha256": cm.sha256(ROOT / "uv.lock"),
                    **cm.versions("fugashi", "ipadic"), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, **res}
    OUT_DIR.mkdir(parents=True)
    with (OUT_DIR / "work2.json").open("x") as fh:
        fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    for a, ws in res["authors"].items():
        print(a, [(w["work_id"], w["sentences"], round(w["matched_share"], 3), w["clipped"]) for w in ws])


if __name__ == "__main__":
    main()
