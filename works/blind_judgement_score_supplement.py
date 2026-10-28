"""Work 2, a supplement to the scoring of the blind judgement: the answers of the researcher, scored after score.json was seen.

The reply of the researcher was excluded from results/blind_judgement/score.json by R1, for one cell outside the cells to
fill in (A2 of the sheet 説明) that was empty in the sent spreadsheet and holds an empty string in the reply. This script,
written after score.json was seen and decided by the researcher on 2026-10-28, reads an empty string as an empty cell and
otherwise applies R1, R2, R3 and the scoring of works/blind_judgement_score.py unchanged. It does not touch the decision,
which rests on the answers of harada only.

Usage:  uv run --with openpyxl==3.1.5 python works/blind_judgement_score_supplement.py        (refuses to overwrite)
Input:  results/blind_judgement/score.json, the sent spreadsheet, the sealed key and the reply of the researcher
Output: results/blind_judgement/score-researcher-supplement.json
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc (ignored by git)
sys.path.insert(0, str(ROOT / "works"))

import _common as cm  # noqa: E402
import blind_judgement_score as bs  # noqa: E402

SCORE = ROOT / "results" / "blind_judgement" / "score.json"
OUT = ROOT / "results" / "blind_judgement" / "score-researcher-supplement.json"
JUDGE = "researcher"


def cells_empty_string_as_empty(path: Path) -> dict[str, dict[tuple[int, int], object]]:
    """bs.cells with an empty string read as an empty cell (the only change of the reading)."""
    return {s: {k: v for k, v in c.items() if v != ""} for s, c in _cells(path).items()}


_cells = bs.cells


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    c = json.loads(bs.COND.read_bytes())
    jc = json.loads(bs.JUDGE_COND.read_bytes())
    inp = c["inputs"]
    if JUDGE == c["deciding_judge"]:
        raise SystemExit("the supplement never scores the deciding judge")
    record_path = ROOT / inp["record"]
    cm.same_as_head([Path(__file__), Path(bs.__file__), bs.COND, bs.JUDGE_COND, SCORE, record_path,
                     ROOT / "works" / "_common.py"])
    state = cm.git_state(["works", "results/blind_judgement"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    score = json.loads(SCORE.read_bytes())
    record = json.loads(record_path.read_bytes())
    sent, reply = ROOT / inp["sent"], ROOT / inp["replies"][JUDGE]
    excluded = score["excluded"].get(JUDGE)
    if excluded is None or excluded["reply_sha256"] != cm.sha256(reply):
        raise SystemExit("the reply is not the one excluded in score.json")
    if excluded["problems"] != ["説明: cells changed at [((2, 1), '')]"]:
        raise SystemExit(f"score.json excluded the reply for other reasons: {excluded['problems']}")
    key_path = ROOT / inp["key"]
    if cm.sha256(sent) != record["spreadsheet_sha256"] or cm.sha256(key_path) != record["key_sha256"]:
        raise SystemExit("the sent spreadsheet or the key differs from the record")
    key = json.loads(key_path.read_bytes())
    if not key["conditions_sha256"] == record["meta"]["conditions_sha256"] == cm.sha256(bs.JUDGE_COND):
        raise SystemExit("the key, the record and the conditions of the judgement do not agree")
    items = key["items"]
    if items != score["key"]:
        raise SystemExit("the key and score.json do not agree on the items")
    bs.cells = cells_empty_string_as_empty
    try:
        problems = bs.check_reply(sent, reply, len(items))
    finally:
        bs.cells = _cells
    if problems:
        raise SystemExit(f"the reply still differs from the sent spreadsheet: {problems}")
    answers, comments = bs.read_answers(reply, len(items))
    versions, rows, seeds = list(jc["versions"]), jc["rows"], jc["common_seeds"]
    judged = bs.score_judge(items, answers, versions, rows, seeds)
    harada = {int(k): v for k, v in score["answers"]["harada"].items()}
    res = {"meta": {"purpose": "the researcher's answers, scored after score.json was seen; not blind (the researcher "
                               "saw the earlier sheet and made the stimuli); descriptive only, not part of the decision",
                    "reading_changed": "an empty string outside the cells to fill in is read as an empty cell",
                    "score_sha256": cm.sha256(SCORE), "script_sha256": cm.sha256(Path(__file__)),
                    "score_script_sha256": cm.sha256(Path(bs.__file__)), "conditions_sha256": cm.sha256(bs.COND),
                    "reply_sha256": cm.sha256(reply), **cm.versions("numpy", "pillow", "openpyxl"), **state,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "judge": {**judged, "comments": comments},
           "controls_pass": judged["controls"]["pass"],
           "rows_remaining": ({v: [r for r in rows if judged["test"][v][r]["remains"]] for v in versions}
                              if judged["controls"]["pass"] else None),
           "agreement_with_harada": bs.agreement(items, harada, answers),
           "answers": {str(k): v for k, v in answers.items()}}
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    with OUT.open("x") as fh:
        fh.write(text)
    print(json.dumps({k: res[k] for k in ("controls_pass", "rows_remaining", "agreement_with_harada")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
