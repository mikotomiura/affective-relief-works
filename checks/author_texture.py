"""Checks 4 and 5 of the overall inspection: the fluctuation of the authors of work 2 under other windows, and the
frequency of their matched words.

Usage:  uv run python checks/author_texture.py            (refuses to overwrite)
Conditions: checks/conditions/overall_inspection.json, check4_fluctuation and check5_frequency, and the inputs of
        works/conditions/work2.json (committed before this script was run)
Input:  the 23 Aozora Bunko zips, MEmoLon, the map B and the MeCab dictionary of work 2 (external/, results/)
Output: results/overall_inspection/author_texture.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import window  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "checks" / "conditions" / "overall_inspection.json"
WORK2_COND = ROOT / "works" / "conditions" / "work2.json"
WORK2_RESULT = ROOT / "results" / "work2" / "work2.json"
OUT_DIR = ROOT / "results" / "overall_inspection"
OUT = OUT_DIR / "author_texture.json"
SEED, PERMUTATIONS, TOP, PRIOR_TOTAL = 20261028, 10000, 30, 1000.0
OPENERS = ("「", "『")


def fluctuation(sentences: list[window.Sentence], w: int) -> dict[str, float]:
    pts = [p for p in window.trajectory(sentences, w) if not p.gap]
    return {"V": float(np.std([p.V for p in pts])), "A": float(np.std([p.A for p in pts])), "windows": len(pts)}


def f_statistic(x: np.ndarray, groups: np.ndarray) -> float:
    m, labels = x.mean(), np.unique(groups)
    between = sum(len(x[groups == g]) * (x[groups == g].mean() - m) ** 2 for g in labels) / (len(labels) - 1)
    within = sum(((x[groups == g] - x[groups == g].mean()) ** 2).sum() for g in labels) / (len(x) - len(labels))
    return float(between / within)


def log_odds(own: Counter, rest: Counter, prior: dict[str, float]) -> dict[str, float]:
    """z-scores of the weighted log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008)."""
    n1, n2, a0 = sum(own.values()), sum(rest.values()), sum(prior.values())
    z = {}
    for w, a in prior.items():
        y1, y2 = own[w], rest[w]
        d = np.log((y1 + a) / (n1 + a0 - y1 - a)) - np.log((y2 + a) / (n2 + a0 - y2 - a))
        z[w] = float(d / np.sqrt(1 / (y1 + a) + 1 / (y2 + a)))
    return z


def jsd(p: Counter, q: Counter) -> float:
    vocab = sorted(set(p) | set(q))
    a = np.array([p[w] for w in vocab], float)
    b = np.array([q[w] for w in vocab], float)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x: np.ndarray) -> float:
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / m[nz])).sum())

    return (kl(a) + kl(b)) / 2


def between_minus_within(D: np.ndarray, labels: np.ndarray) -> float:
    same = labels[:, None] == labels[None, :]
    iu = np.triu_indices(len(labels), 1)
    return float(D[iu][~same[iu]].mean() - D[iu][same[iu]].mean())


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    c4 = c["check4_fluctuation"]
    _, w2 = cm.load_conditions(WORK2_COND)
    cm.same_as_head([COND, WORK2_COND, Path(__file__), WORK2_RESULT, ROOT / w2["map"]["file"]])
    state = cm.git_state(["checks", "src", "works", "results/work2", "results/calibrate_memolon.json",
                          "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of work 2")
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if lex.sha256 != w2["lexicon"]["sha256"]:
        raise SystemExit("the lexicon differs from that of work 2")
    vmap = calibration_map(cm.check_file(ROOT / w2["map"]["file"], w2["map"]["sha256"], "the calibration"))
    done = {w["work_id"]: w for ws in json.loads(WORK2_RESULT.read_text())["authors"].values() for w in ws}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    works = []
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
        matched = [p for p in t.points if not p.unmatched]
        if len(t.points) != ref["sentences"] or len(matched) / len(t.points) != ref["matched_share"]:
            raise SystemExit(f"{w['work_id']}: the text step gives another result than results/work2")
        sents = [window.Sentence(p.V, p.A, not p.unmatched) for p in t.points]
        works.append({"author": w["author"], "work_id": w["work_id"], "title": w["title"],
                      "fluctuation": {str(W): fluctuation(sents, W) for W in c4["windows"]},
                      "length": len(t.points),
                      "dialogue": sum(s.startswith(OPENERS) for s in t.sentences) / len(t.sentences),
                      "matched_share": len(matched) / len(t.points),
                      "z_mean": float(np.mean([p.Z for p in matched])),
                      "words": Counter(x for p in t.points for x in p.matched)})

    authors = sorted({w["author"] for w in works}, key=[w["author"] for w in works].index)
    labels = np.array([authors.index(w["author"]) for w in works])
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(labels) for _ in range(PERMUTATIONS)]

    # check 4
    per_author = {a: {str(W): {d: float(np.mean([w["fluctuation"][str(W)][d] for w in works if w["author"] == a]))
                               for d in ("V", "A")} for W in c4["windows"]} for a in authors}
    order = {str(W): [per_author[a][str(W)]["V"] for a in authors] for W in c4["windows"]}
    taus = {f"{a}-{b}": float(stats.kendalltau(order[str(a)], order[str(b)]).statistic)
            for a, b in itertools.combinations(c4["windows"], 2)}
    x40 = np.array([w["fluctuation"]["40"]["V"] for w in works])
    f_obs = f_statistic(x40, labels)
    f_null = np.array([f_statistic(x40, p) for p in perms])
    rhos = {k: float(stats.spearmanr(x40, [w[k] for w in works]).statistic)
            for k in ("length", "dialogue", "matched_share", "z_mean")}
    check4 = {"per_author": per_author, "kendall_tau_V": taus, "F": f_obs,
              "F_p": float((1 + np.sum(f_null >= f_obs)) / (PERMUTATIONS + 1)), "spearman_with_V_fluctuation": rhos,
              "criteria": {"R1": taus["20-80"] >= 0.6, "R2": float((1 + np.sum(f_null >= f_obs)) / (PERMUTATIONS + 1)) < 0.05,
                           "R3": all(abs(r) < 0.5 for r in rhos.values()),
                           "R3_confounds": [k for k, r in rhos.items() if abs(r) >= 0.5]}}

    # check 5
    by_author = {a: sum((w["words"] for w in works if w["author"] == a), Counter()) for a in authors}
    total = sum(by_author.values(), Counter())
    n_total = sum(total.values())
    prior = {w: PRIOR_TOTAL * n / n_total for w, n in total.items()}

    def va(word: str) -> tuple[float, float]:
        return vmap(*lex.entries[word])

    distinctive, mean_v, top_v = {}, {}, {}
    for a in authors:
        rest = sum((by_author[b] for b in authors if b != a), Counter())
        z = log_odds(by_author[a], rest, prior)
        top = sorted(z, key=lambda w: (-z[w], w))[:TOP]
        distinctive[a] = [{"word": w, "z": z[w], "count": by_author[a][w], "V": va(w)[0], "A": va(w)[1]} for w in top]
        n = sum(by_author[a].values())
        mean_v[a] = float(sum(cnt * va(w)[0] for w, cnt in by_author[a].items()) / n)
        top_v[a] = float(np.mean([d["V"] for d in distinctive[a]]))
    D = np.array([[jsd(p["words"], q["words"]) if i != j else 0.0 for j, q in enumerate(works)]
                  for i, p in enumerate(works)])
    bw_obs = between_minus_within(D, labels)
    bw_null = np.array([between_minus_within(D, p) for p in perms])
    same = labels[:, None] == labels[None, :]
    iu = np.triu_indices(len(works), 1)
    ratio = (max(top_v.values()) - min(top_v.values())) / (max(mean_v.values()) - min(mean_v.values()))
    check5 = {"token_weighted_mean_V": mean_v, "distinctive_mean_V": top_v, "range_ratio": float(ratio),
              "jsd_between_mean": float(D[iu][~same[iu]].mean()), "jsd_within_mean": float(D[iu][same[iu]].mean()),
              "jsd_p": float((1 + np.sum(bw_null >= bw_obs)) / (PERMUTATIONS + 1)), "distinctive": distinctive,
              "criteria": {"Q1": ratio >= 2.0, "Q2": float((1 + np.sum(bw_null >= bw_obs)) / (PERMUTATIONS + 1)) < 0.05,
                           "Q3": "by eye, recorded in the decisions"}}

    for w in works:
        w["words"] = {"tokens": sum(w["words"].values()), "types": len(w["words"])}
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                    "work2_conditions_sha256": cm.sha256(WORK2_COND), "lexicon_sha256": lex.sha256,
                    "dicdir_sha256": tagger.dicdir_sha256, **cm.versions("fugashi", "ipadic"),
                    "numpy": np.__version__, "scipy": scipy.__version__,
                    "python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                    **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "permutations": PERMUTATIONS, "seed": SEED, "works": works, "check4": check4, "check5": check5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"  # built before the file is opened
    with OUT.open("x") as fh:
        fh.write(text)
    print(json.dumps({k: v for k, v in check4.items() if k != "per_author"}, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in check5.items() if k != "distinctive"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
