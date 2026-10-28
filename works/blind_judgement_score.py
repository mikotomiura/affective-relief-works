"""Work 2, the scoring of the blind judgement: the rule of the conditions applied to the answers of the two judges.

Usage:  uv run --with openpyxl==3.1.5 python works/blind_judgement_score.py        (refuses to overwrite)
Conditions: works/conditions/blind_judgement_score.json and works/conditions/blind_judgement.json (committed before any
        answer was seen)
Input:  the sent spreadsheet, the sealed key and the two replies (advice/blind-judgement/, outside git)
Output: results/blind_judgement/score.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc (ignored by git)
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import _common as cm  # noqa: E402
from blind_judgement import KINDS, anchored_images  # noqa: E402

COND = ROOT / "works" / "conditions" / "blind_judgement_score.json"
JUDGE_COND = ROOT / "works" / "conditions" / "blind_judgement.json"
OUT = ROOT / "results" / "blind_judgement" / "score.json"
FIRST_ROW, ANSWER_COL, COMMENT_COL = 2, 3, 4
DIFFERENT, SAME, UNSURE = "違って見える", "同じに見える", "わからない"


# ---------------------------------------------------------------- reading


def normalise(v) -> str | None:
    """An answer as given: one of the three choices, or None when empty. Anything else is an error."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    if s not in (DIFFERENT, SAME, UNSURE):
        raise ValueError(f"not one of the choices: {s!r}")
    return s


def cells(path: Path) -> dict[str, dict[tuple[int, int], object]]:
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=False)
    return {ws.title: {(c.row, c.column): c.value for row in ws.iter_rows() for c in row if c.value is not None}
            for ws in wb}


def anchored_pixels(path: Path) -> dict[int, str]:
    """Row (1-based) -> SHA-256 of the decoded RGB pixels of the image anchored in column B of that row, for any kind of
    anchor Excel may write (one-cell or two-cell)."""
    with zipfile.ZipFile(path) as z:
        out: dict[int, str] = {}
        for d in (n for n in z.namelist() if re.fullmatch(r"xl/drawings/drawing\d+\.xml", n)):
            rels = z.read(d.replace("drawings/", "drawings/_rels/") + ".rels").decode("utf-8")
            target = {}
            for rel in re.findall(r"<Relationship\b[^>]*>", rels):
                rid, tgt = re.search(r'Id="(rId\d+)"', rel).group(1), re.search(r'Target="([^"]+)"', rel).group(1)
                target[rid] = "xl/media/" + tgt.rsplit("/media/", 1)[1]
            xml = z.read(d).decode("utf-8")
            for anchor in re.findall(r"<(?:xdr:)?(?:one|two)CellAnchor\b.*?</(?:xdr:)?(?:one|two)CellAnchor>", xml,
                                     flags=re.S):
                frm = re.search(r"<(?:xdr:)?from>(.*?)</(?:xdr:)?from>", anchor, flags=re.S).group(1)
                col = int(re.search(r"<(?:xdr:)?col>(\d+)</(?:xdr:)?col>", frm).group(1))
                row = int(re.search(r"<(?:xdr:)?row>(\d+)</(?:xdr:)?row>", frm).group(1)) + 1
                m = re.search(r'r:embed="(rId\d+)"', anchor)
                if col != 1 or m is None:
                    continue
                if row in out:
                    raise ValueError(f"two images on row {row}")
                px = np.asarray(Image.open(io.BytesIO(z.read(target[m.group(1)]))).convert("RGB"))
                out[row] = hashlib.sha256(px.tobytes() + str(px.shape).encode()).hexdigest()
    return out


def check_reply(sent: Path, reply: Path, n_items: int) -> list[str]:
    """R1 and R2: what differs between the reply and the sent spreadsheet outside the cells to fill in."""
    problems = []
    a, b = cells(sent), cells(reply)
    if list(a) != list(b):
        problems.append(f"sheets {list(a)} became {list(b)}")
    fill = {(r, c) for r in range(FIRST_ROW, FIRST_ROW + n_items) for c in (ANSWER_COL, COMMENT_COL)}
    for sheet in a:
        x = {k: v for k, v in a[sheet].items() if not (sheet == "判定" and k in fill)}
        y = {k: v for k, v in b.get(sheet, {}).items() if not (sheet == "判定" and k in fill)}
        if x != y:
            problems.append(f"{sheet}: cells changed at {sorted(set(x.items()) ^ set(y.items()))[:5]}")
    if anchored_pixels(sent) != anchored_pixels(reply):
        problems.append("the images differ from the sent ones")
    return problems


def read_answers(reply: Path, n_items: int) -> tuple[dict[int, str | None], dict[int, str]]:
    """R3: item number -> answer (None when empty) and item number -> comment."""
    c = cells(reply)["判定"]
    answers, comments = {}, {}
    for n in range(1, n_items + 1):
        r = FIRST_ROW + n - 1
        if c.get((r, 1)) != n:
            raise ValueError(f"row {r} does not hold item {n}")
        try:
            answers[n] = normalise(c.get((r, ANSWER_COL)))
        except ValueError as e:
            raise ValueError(f"item {n}: {e}") from None
        if c.get((r, COMMENT_COL)) is not None:
            comments[n] = str(c[(r, COMMENT_COL)])
    return answers, comments


# ---------------------------------------------------------------- scoring


def score_judge(items: list[dict], answers: dict[int, str | None], versions: list[str], rows: list[str],
                seeds: list[int]) -> dict:
    by_kind = lambda k: [(it["number"], it) for it in items if it["kind"] == k]  # noqa: E731
    counts_of = {k: len(by_kind(k)) for k in KINDS}
    if counts_of != {"test": len(versions) * len(rows) * len(seeds), "same_image": len(rows), "seed_differs": len(rows),
                     "set_differs": 5, "repeat": len(rows)}:
        raise ValueError(f"unexpected counts of items in the key: {counts_of}")
    same = [answers[n] for n, _ in by_kind("same_image")]
    diff = [answers[n] for n, _ in by_kind("seed_differs")]
    controls = {"same_image": same, "seed_differs": diff,
                "pass": all(x == SAME for x in same) and sum(x == DIFFERENT for x in diff) >= 4}
    table: dict[str, dict[str, dict]] = {}
    for v in versions:
        table[v] = {}
        for r in rows:
            by_seed = {}
            for n, it in by_kind("test"):
                if it["top"]["version"] == v and it["top"]["row"] == r:
                    by_seed[str(it["top"]["seed"])] = answers[n]
            if sorted(by_seed) != sorted(str(s) for s in seeds):
                raise ValueError(f"{v} {r}: test items for seeds {sorted(by_seed)}")
            table[v][r] = {"answers": by_seed, "remains": all(x == DIFFERENT for x in by_seed.values())}
    counts = {v: {k: sum(1 for n, it in by_kind("test") if it["top"]["version"] == v and answers[n] == k)
                  for k in (DIFFERENT, SAME, UNSURE)} for v in versions}
    for v in versions:
        counts[v]["empty"] = sum(1 for n, it in by_kind("test") if it["top"]["version"] == v and answers[n] is None)
    repeats = [{"item": n, "first": it["repeat_of"], "answer_first": answers[it["repeat_of"]], "answer_repeat": answers[n],
                "agree": answers[n] == answers[it["repeat_of"]]} for n, it in by_kind("repeat")]
    set_diff = [{"item": n, "rows": [it["top"]["row"], it["bottom"]["row"]], "version": it["top"]["version"],
                 "answer": answers[n]} for n, it in by_kind("set_differs")]
    return {"controls": controls, "test": table, "counts": counts, "repeats": repeats, "set_differs": set_diff,
            "empty_total": sum(1 for x in answers.values() if x is None)}


def decide(h: dict, rows: list[str]) -> dict:
    if not h["controls"]["pass"]:
        return {"decided": False, "result": "this judgement cannot decide (the controls of the deciding judge did not pass)"}
    remain = {v: [r for r in rows if h["test"][v][r]["remains"]] for v in h["test"]}
    return {"decided": True, "rows_remaining": remain, "two_claims": bool(remain["G0"])}


def agreement(items: list[dict], a: dict[int, str | None], b: dict[int, str | None]) -> dict:
    tests = [it["number"] for it in items if it["kind"] == "test"]
    return {"test_items": len(tests), "answered_alike": sum(a[n] == b[n] for n in tests)}


# ---------------------------------------------------------------- main


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    jc = json.loads(JUDGE_COND.read_bytes())
    inp = c["inputs"]
    record_path = ROOT / inp["record"]
    cm.same_as_head([COND, JUDGE_COND, Path(__file__), ROOT / "works" / "_common.py", record_path])
    state = cm.git_state(["works", "results/blind_judgement"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    record = json.loads(record_path.read_bytes())
    sent, key_path = ROOT / inp["sent"], ROOT / inp["key"]
    if cm.sha256(sent) != record["spreadsheet_sha256"] or cm.sha256(key_path) != record["key_sha256"]:
        raise SystemExit("the sent spreadsheet or the key differs from the record")
    key = json.loads(key_path.read_bytes())
    if not key["conditions_sha256"] == record["meta"]["conditions_sha256"] == cm.sha256(JUDGE_COND):
        raise SystemExit("the key, the record and the conditions of the judgement do not agree")
    if anchored_images(sent) != {it["number"] + 1: it["image_sha256"] for it in key["items"]}:  # B4b again
        raise SystemExit("the images of the sent spreadsheet are not those of the key")
    items = key["items"]
    n = len(items)
    versions, rows, seeds = list(jc["versions"]), jc["rows"], jc["common_seeds"]
    judges, excluded = {}, {}
    for judge, rel in inp["replies"].items():
        reply = ROOT / rel
        problems = check_reply(sent, reply, n)
        if problems and judge == c["deciding_judge"]:
            raise SystemExit(f"{judge}: the reply differs from the sent spreadsheet: {problems}")
        if problems:  # the other judge cannot block the deciding one; the reply is reported, not scored
            excluded[judge] = {"reply_sha256": cm.sha256(reply), "problems": problems}
            continue
        answers, comments = read_answers(reply, n)
        judges[judge] = {"reply_sha256": cm.sha256(reply), **score_judge(items, answers, versions, rows, seeds),
                         "comments": comments, "_answers": answers}
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                    **cm.versions("numpy", "pillow", "openpyxl"), **state,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "decision": decide(judges["harada"], rows),
           "agreement": (agreement(items, judges["harada"]["_answers"], judges["researcher"]["_answers"])
                         if "researcher" in judges else None),
           "excluded": excluded,
           "judges": {j: {k: v for k, v in d.items() if k != "_answers"} for j, d in judges.items()},
           "answers": {j: {str(k): v for k, v in d["_answers"].items()} for j, d in judges.items()},
           "key": items}
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    with OUT.open("x") as fh:
        fh.write(text)
    print(json.dumps(res["decision"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
