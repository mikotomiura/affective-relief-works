"""Check 2 of the overall inspection: which open quantity of the field lets the seed change the look of a tile.

Usage:  uv run python checks/content_channel.py            (refuses to overwrite)
Conditions: checks/conditions/overall_inspection.json, check2_content_channel (committed before this script was run)
Input:  results/work1/work1.json (V and A of three sentences)
Output: results/overall_inspection/content_channel.json and content_channel.png
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

import _common as cm  # noqa: E402
from atact.field import RenderConfig, procedural_field  # noqa: E402
from atact.mapping import params  # noqa: E402
from atact.transform import t_gamma  # noqa: E402

COND = ROOT / "checks" / "conditions" / "overall_inspection.json"
OUT_DIR = ROOT / "results" / "overall_inspection"
OUT = OUT_DIR / "content_channel.json"
FIG = OUT_DIR / "content_channel.png"
POINTS = ROOT / "results" / "work1" / "work1.json"
N, GAP, SENTENCE_GAP = 180, 6, 24
D_MM, S_MAX = 1.2, 0.3


def pick(pts: list[dict]) -> list[int]:
    """Smallest s, the median s (the lower middle value, ties to the smaller index) and the largest s."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i]["s"], i))
    return [order[0], order[(len(order) - 1) // 2], order[-1]]


def heights(V: float, A: float, seed: int, form: str, warp: float) -> np.ndarray:
    f, s, g = (float(x) for x in params(V, A))
    cfg = RenderConfig(n=N, octaves=4, normalize="ampsum", freq_unit="cycles", octave_form=form, warp=warp)
    return s * t_gamma(procedural_field(f, cfg, seed), g)


def common_grey(H: np.ndarray) -> np.ndarray:
    return D_MM * H / (2 * S_MAX) / D_MM + 0.5


def main() -> None:
    if OUT.exists() or FIG.exists():
        raise SystemExit(f"{OUT} or {FIG} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)["check2_content_channel"]
    if cm.installed_commit() != json.loads(raw)["generator"]["commit"]:
        raise SystemExit("atact is not installed from the commit of the conditions")
    cm.same_as_head([COND, Path(__file__), POINTS])
    state = cm.git_state(["checks", "src", "works", "results/work1", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pts = json.loads(POINTS.read_text())["sentences"]
    idx = pick(pts)
    seeds = c["seeds"]
    rows, measures = [], {}
    for name, v in c["variants"].items():
        Hs = {(i, sd): heights(pts[i]["V"], pts[i]["A"], sd, v["octave_form"], v["warp"]) for i in idx for sd in seeds}
        corr = [np.mean([abs(np.corrcoef(Hs[i, a].ravel(), Hs[i, b].ravel())[0, 1])
                         for a, b in itertools.combinations(seeds, 2)]) for i in idx]
        sds = [np.mean([Hs[i, sd].std() for sd in seeds]) for i in idx]
        measures[name] = {"mean_abs_corr_between_seeds": float(np.mean(corr)),
                          "per_sentence_abs_corr": [float(x) for x in corr],
                          "sd_of_H_per_sentence": [float(x) for x in sds],
                          "affect_spread": float((max(sds) - min(sds)) / np.mean(sds))}
        row = []
        for j, i in enumerate(idx):
            if j:
                row.append(np.ones((N, SENTENCE_GAP)))
            for k, sd in enumerate(seeds):
                if k:
                    row.append(np.ones((N, GAP)))
                row.append(np.flipud(np.clip(common_grey(Hs[i, sd]), 0.0, 1.0)))
        rows.append(np.hstack(row))
    sep = np.ones((GAP * 3, rows[0].shape[1]))
    canvas = np.vstack([x for r in rows for x in (r, sep)][:-1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(canvas * 255).astype(np.uint8), mode="L").save(FIG)
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                    "points_sha256": cm.sha256(POINTS), "atact_commit": cm.installed_commit(),
                    "numpy": np.__version__, "python": platform.python_version(),
                    "platform": f"{platform.system()} {platform.machine()}", **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "sentences": [{"i": i, "text": pts[i]["text"], "V": pts[i]["V"], "A": pts[i]["A"], "s": pts[i]["s"]}
                         for i in idx],
           "seeds": seeds, "rows": list(c["variants"]), "measures": measures,
           "figure": {"file": str(FIG.relative_to(ROOT)), "sha256": cm.sha256(FIG)}}
    text = json.dumps(res, indent=1, ensure_ascii=False) + "\n"  # built before the file is opened
    with OUT.open("x") as fh:
        fh.write(text)
    for k, m in measures.items():
        print(k, round(m["mean_abs_corr_between_seeds"], 3), round(m["affect_spread"], 3))


if __name__ == "__main__":
    main()
