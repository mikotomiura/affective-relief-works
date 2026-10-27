"""Fix the common stretch factor k of work 2 from 40 reference works of other authors (no candidate of work 2).

Usage:  uv run python checks/reference_stretch.py      (refuses to overwrite)
Conditions: checks/conditions/reference_stretch.json (committed before this script was run)
Input:  the Aozora Bunko list (CSV), the zips of the drawn works (fetched, kept in external/), the Japanese MEmoLon
        lexicon (external/) and results/calibrate_memolon.json (map B)
Output: results/reference_stretch.json
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import stretch  # noqa: E402
import window  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "checks" / "conditions" / "reference_stretch.json"
OUT = ROOT / "results" / "reference_stretch.json"
ZIPS = ROOT / "external" / "reference-stretch-20261028" / "zips"
SCRIPT = Path(__file__).resolve()
CODE = ["checks/reference_stretch.py", "works/_common.py", "src/pipeline.py", "src/english_text.py",
        "src/japanese_text.py", "src/window.py", "src/stretch.py", "src/relief.py"]
PCTS = (1, 50, 99)


def eligible(rows: list[dict], el: dict) -> dict[str, list[dict]]:
    """Eligible works by 人物ID (one author row per work, no excluded person in any row of the work)."""
    excluded = set(el["excluded_persons"])
    by_work: dict[str, list[dict]] = {}
    for r in rows:
        by_work.setdefault(r["作品ID"], []).append(r)
    out: dict[str, list[dict]] = {}
    for wid, rs in by_work.items():
        if any(r["人物ID"] in excluded for r in rs):
            continue
        authors = [r for r in rs if r["役割フラグ"] == "著者"]
        if len(authors) != 1:
            continue
        r = authors[0]
        if (r["文字遣い種別"] == "新字新仮名" and r["作品著作権フラグ"] == "なし" and r["分類番号"].strip() == "NDC 913"
                and r["テキストファイル符号化方式"] == "ShiftJIS" and r["テキストファイルURL"].endswith(".zip")):
            out.setdefault(r["人物ID"], []).append(r)
    return {p: sorted(ws, key=lambda r: r["作品ID"]) for p, ws in out.items()}


def fetch_url(url: str) -> bytes:
    """The zip at url; any failure (network, HTTP status, a body that is not a zip) stops the run."""
    time.sleep(1.0)
    req = urllib.request.Request(url, headers={"User-Agent": "research-fetch (reference works for a stretch factor)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as fh:
            data = fh.read()
    except OSError as e:
        raise SystemExit(f"fetching {url} failed ({e}); run again to continue with the zips kept so far") from e
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise SystemExit(f"{url} did not return a zip; run again later")
    return data


def draw(works: dict[str, list[dict]], dr: dict, rng_range: list[int], zips: Path,
         fetch: Callable[[str], bytes]) -> tuple[list[dict], list[dict]]:
    """The drawn works (with their decoded text) and the skipped ones, by the rule of the conditions."""
    rng = np.random.default_rng(dr["seed"])
    persons = sorted(works)
    chosen, skipped = [], []
    lo, hi = rng_range
    for pi in rng.permutation(len(persons)):
        if len(chosen) == dr["n_works"]:
            break
        ws = works[persons[pi]]
        for wi in rng.permutation(len(ws)):
            r = ws[wi]
            url = r["テキストファイルURL"]
            rec = {"person": r["人物ID"], "author": r["姓"] + r["名"], "work_id": r["作品ID"], "title": r["作品名"],
                   "url": url}
            zp = zips / r["人物ID"] / url.rsplit("/", 1)[1]
            if not zp.exists():
                # a failed request stops the run (it must not change the draw); a rerun uses the zips kept so far
                data = fetch(url)
                zp.parent.mkdir(parents=True, exist_ok=True)
                tmp = zp.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.rename(zp)
            rec["zip"] = str(zp.relative_to(ROOT)) if zp.is_relative_to(ROOT) else zp.name
            rec["zip_sha256"] = cm.sha256(zp)
            try:  # only the content of the file may skip a work (the same bytes always give the same outcome)
                rec["txt_name"], raw = read_aozora_zip(zp)
                text = raw.decode("cp932")
                body, _, note, kept, _ = work2_body(text)
            except (zipfile.BadZipFile, UnicodeDecodeError, ValueError) as e:  # recorded, not hidden
                skipped.append({**rec, "reason": f"{type(e).__name__}: {e}"[:200]})
                continue
            if not lo <= len(kept) <= hi:
                skipped.append({**rec, "reason": f"{len(kept)} sentences"})
                continue
            chosen.append({**rec, "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                           "sentences": len(kept), "date_note": note, "text": text})
            break
    if len(chosen) < dr["n_works"]:
        raise SystemExit(f"only {len(chosen)} works could be drawn")
    return chosen, skipped


def nearest(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    return {str(q): xs[round(q / 100 * (len(xs) - 1))] for q in PCTS}


def summary(c: dict, chosen: list[dict], lex, tagger, vmap) -> dict:
    trajs, per = [], []
    for w in chosen:
        t = work2_text(w.pop("text"), lex, tagger, vmap)
        if t.body_sha256 != w["body_sha256"] or len(t.sentences) != w["sentences"]:
            raise SystemExit(f"{w['work_id']}: the text step differs between the draw and the computation")
        have = [p for p in t.trajectory if not p.gap]
        if not have:
            raise SystemExit(f"{w['work_id']}: no window with a point")
        trajs.append(t.trajectory)
        matched = sum(not p.unmatched for p in t.points)
        per.append({**w, "matched_share": matched / len(t.points),
                    "matched_token_share": sum(p.Z for p in t.points) / max(1, sum(p.n_candidates for p in t.points)),
                    "windows": len(t.trajectory), "gaps": len(t.trajectory) - len(have),
                    "V": nearest([p.V for p in have]), "A": nearest([p.A for p in have]),
                    "max_abs": nearest([max(abs(p.V), abs(p.A)) for p in have])})
    kr = c["k_rule"]
    k, Q = stretch.factor(trajs, kr["target"], kr["quantile"])
    pooled = {}
    for name, f in (("unstretched", 1.0), ("stretched", k)):
        pts = [stretch.stretch(t, f).points for t in trajs]
        vals = {"V": [], "A": []}
        wts = []
        for ps in pts:
            have = [p for p in ps if not p.gap]
            vals["V"] += [p.V for p in have]
            vals["A"] += [p.A for p in have]
            wts += [1.0 / len(have)] * len(have)
        pooled[name] = {d: {str(q): stretch.weighted_quantile(v, wts, q / 100) for q in PCTS} for d, v in vals.items()}
    loo = [stretch.factor(trajs[:i] + trajs[i + 1 :], kr["target"], kr["quantile"])[0] for i in range(len(trajs))]
    return {"result": {"k": k, "Q": Q, "target": kr["target"], "quantile": kr["quantile"],
                       "clip_rate": stretch.clip_rate(trajs, k), "pooled_percentiles": pooled,
                       "k_leave_one_out": {"min": min(loo), "max": max(loo),
                                           "work_of_min": per[loo.index(min(loo))]["work_id"],
                                           "work_of_max": per[loo.index(max(loo))]["work_id"]}},
            "works": per}


def validate(c: dict) -> None:
    """The settings that the code fixes must be those written in the conditions."""
    wd, st, kr = c["window"], c["step"], c["k_rule"]
    if (cm.need(wd, "W", int, "window.") != window.W_WORK2
            or cm.need(wd, "min_matched_fraction", float, "window.") != window.MIN_MATCHED_FRACTION):
        raise SystemExit("the window differs from that of src/window.py")
    if st.get("select") != "pos" or st.get("order") != "word":
        raise SystemExit("the step is select 'pos', order 'word'")
    if cm.need(kr, "target", float, "k_rule.") != stretch.TARGET or cm.need(kr, "quantile", float, "k_rule.") != \
            stretch.QUANTILE:
        raise SystemExit("the rule for k differs from src/stretch.py")
    lo, hi = c["eligible"]["sentences"]
    if not (isinstance(lo, int) and isinstance(hi, int) and 0 < lo <= hi):
        raise SystemExit("sentences must be a pair of whole numbers")
    if cm.need(c["draw"], "n_works", int, "draw.") < 2 or cm.need(c["draw"], "seed", int, "draw.") < 0:
        raise SystemExit("draw: n_works at least 2 and a seed")


def excluded_in_rows(rows: list[dict], chosen: list[dict], excluded: set[str]) -> list[str]:
    """Work IDs of the chosen works in which an excluded person appears in any row of the list (a check made on the
    rows again, apart from eligible)."""
    ids = {w["work_id"] for w in chosen}
    return sorted({r["作品ID"] for r in rows if r["作品ID"] in ids and r["人物ID"] in excluded})


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    validate(c)
    cm.same_as_head([COND, ROOT / c["map"]["file"], *(ROOT / p for p in CODE)])
    state = cm.git_state(["checks", "works", "src", "results/calibrate_memolon.json", "pyproject.toml", "uv.lock",
                          ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_generator(c["generator"])
    vmap = calibration_map(cm.check_file(ROOT / c["map"]["file"], c["map"]["sha256"], "the calibration"))
    cm.check_versions(c["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != c["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of the conditions")
    lst = cm.check_file(ROOT / c["list"]["file"], c["list"]["sha256"], "the list")
    rows = list(csv.DictReader(io.StringIO(lst.decode("utf-8-sig"), newline="")))
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    works = eligible(rows, c["eligible"])
    ZIPS.mkdir(parents=True, exist_ok=True)
    chosen, skipped = draw(works, c["draw"], c["eligible"]["sentences"], ZIPS, fetch_url)
    bad = excluded_in_rows(rows, chosen, set(c["eligible"]["excluded_persons"]))
    if bad:
        raise SystemExit(f"works with an excluded person were drawn: {bad}")
    lex = japanese_text.load_memolon(ROOT / c["lexicon"]["file"])
    if lex.sha256 != c["lexicon"]["sha256"]:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    res = summary(c, chosen, lex, tagger, vmap)
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(SCRIPT),
                    "lexicon_sha256": lex.sha256, "dicdir_sha256": tagger.dicdir_sha256,
                    "uv_lock_sha256": cm.sha256(ROOT / "uv.lock"), **cm.versions("fugashi", "ipadic", "numpy"),
                    **state, "eligible_persons": len(works), "eligible_works": sum(map(len, works.values())),
                    "started": started, "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           **res, "skipped": skipped}
    with OUT.open("x") as fh:
        fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(res["result"], indent=1), len(skipped), "skipped")


if __name__ == "__main__":
    main()
