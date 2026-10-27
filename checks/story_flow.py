"""Check 3 of the overall inspection: whether the values of work 1 follow an order along the story.

Usage:  uv run python checks/story_flow.py            (refuses to overwrite)
Conditions: checks/conditions/overall_inspection.json, check3_story_flow (committed before this script was run)
Input:  results/work1/work1.json (V, A) and results/strip_sketch.json (the paragraph of every sentence)
Output: results/overall_inspection/story_flow.json and story_flow.png
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

import _common as cm  # noqa: E402

COND = ROOT / "checks" / "conditions" / "overall_inspection.json"
OUT_DIR = ROOT / "results" / "overall_inspection"
OUT = OUT_DIR / "story_flow.json"
FIG = OUT_DIR / "story_flow.png"
POINTS = ROOT / "results" / "work1" / "work1.json"
PARAS = ROOT / "results" / "strip_sketch.json"
SEED, PERMUTATIONS, MA, DPI = 20261028, 10000, 9, 200


def r1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def eta2(x: np.ndarray, groups: np.ndarray) -> float:
    m = x.mean()
    between = sum(len(x[groups == g]) * (x[groups == g].mean() - m) ** 2 for g in np.unique(groups))
    return float(between / ((x - m) ** 2).sum())


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centred mean over w values, over fewer at the ends."""
    h = w // 2
    return np.array([x[max(0, i - h) : i + h + 1].mean() for i in range(len(x))])


def p_value(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def main() -> None:
    if OUT.exists() or FIG.exists():
        raise SystemExit(f"{OUT} or {FIG} exists; refusing to overwrite")
    raw = COND.read_bytes()
    cm.same_as_head([COND, Path(__file__), POINTS, PARAS])
    state = cm.git_state(["checks", "src", "works", "results/work1", "results/strip_sketch.json", "pyproject.toml",
                          "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pts = json.loads(POINTS.read_text())["sentences"]
    sk = json.loads(PARAS.read_text())["sentences"]
    if len(sk) != len(pts) or any(abs(a["V"] - b["V"]) > 0 or abs(a["A"] - b["A"]) > 0 for a, b in zip(sk, pts)):
        raise SystemExit("the strip sketch does not hold the points of work 1")
    para = np.array([s["paragraph"] for s in sk])
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(len(pts)) for _ in range(PERMUTATIONS)]
    stats = {}
    for dim in ("V", "A"):
        x = np.array([p[dim] for p in pts])
        null_r1 = np.array([r1(x[p]) for p in perms])
        null_eta = np.array([eta2(x[p], para) for p in perms])  # the labels stay in place: paragraph sizes kept
        o_r1, o_eta = r1(x), eta2(x, para)
        stats[dim] = {"r1": o_r1, "r1_p": p_value(o_r1, null_r1), "r1_null_95": float(np.quantile(null_r1, 0.95)),
                      "eta2": o_eta, "eta2_p": p_value(o_eta, null_eta),
                      "eta2_null_95": float(np.quantile(null_eta, 0.95))}
    crit = {"F1": any(stats[d]["r1_p"] < 0.05 for d in stats), "F2": any(stats[d]["eta2_p"] < 0.05 for d in stats),
            "F3": "by eye, recorded in the decisions"}
    starts = [0] + [i for i in range(1, len(para)) if para[i] != para[i - 1]]
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True, dpi=DPI)
    for ax, dim in zip(axes, ("V", "A")):
        x = np.array([p[dim] for p in pts])
        for s in starts:
            ax.axvline(s - 0.5, color="0.85", linewidth=0.6, zorder=0)
        ax.plot(x, color="0.65", linewidth=0.8)
        ax.plot(moving_average(x, MA), color="0.1", linewidth=2 * 72 / DPI)  # 2 px
        ax.set_ylabel(dim)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="0.3", labelsize=8)
    axes[-1].set_xlabel("sentence")
    axes[0].set_title("Work 1: V and A by sentence (grey), moving mean of 9 (black), paragraphs (light lines)",
                      fontsize=8, loc="left")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                    "points_sha256": cm.sha256(POINTS), "paragraphs_sha256": cm.sha256(PARAS),
                    "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                    "python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                    **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "permutations": PERMUTATIONS, "seed": SEED, "stats": stats, "criteria": crit,
           "figure": {"file": str(FIG.relative_to(ROOT)), "sha256": cm.sha256(FIG)}}
    text = json.dumps(res, indent=1, ensure_ascii=False) + "\n"  # built before the file is opened
    with OUT.open("x") as fh:
        fh.write(text)
    print(json.dumps(stats, indent=1), crit)


if __name__ == "__main__":
    main()
