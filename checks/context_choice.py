"""Check 6 (a supplement to the overall inspection): whether the choice of an author differs from the other authors
when the context is held fixed. A: sets of variants that say the same thing; B: word frequencies within one scene.

Usage:  uv run python checks/context_choice.py            (refuses to overwrite)
Conditions: checks/conditions/check6_context.json, and the inputs of works/conditions/work2.json (committed before
        this script was run)
Input:  the 23 Aozora Bunko zips, MEmoLon, the map B and the MeCab dictionary of work 2 (external/, results/)
Output: results/check6_context/context_choice.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import fugashi  # noqa: E402
import ipadic  # noqa: E402
import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
from author_texture import between_minus_within, jsd  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "checks" / "conditions" / "check6_context.json"
WORK2_COND = ROOT / "works" / "conditions" / "work2.json"
WORK2_RESULT = ROOT / "results" / "work2" / "work2.json"
OUT_DIR = ROOT / "results" / "check6_context"
OUT = OUT_DIR / "context_choice.json"
CODE = ["checks/context_choice.py", "checks/author_texture.py", "works/_common.py", "src/pipeline.py",
        "src/japanese_text.py"]
SEED, PERMUTATIONS = 20261028, 10000
OPENERS = ("「", "『")

Lemma = tuple[str, str, str]  # (lemma, pos1, pos2)


def lemmas(tagger: fugashi.GenericTagger, sentence: str) -> list[Lemma]:
    return [(w.feature[6], w.feature[0], w.feature[1]) for w in tagger(sentence)]


def match_set(toks: list[Lemma], spec: dict) -> Counter:
    """Counts of the variants of one set in one sentence; a token counts at most once (two-lemma variants first)."""
    pos1, pos2 = set(spec["pos1"]), set(spec.get("pos2", []))
    forms = [(name, tuple(f)) for name, fs in spec["variants"].items() for f in fs]
    used, out = set(), Counter()
    for length in sorted({len(f) for _, f in forms}, reverse=True):
        for i in range(len(toks) - length + 1):
            if any(j in used for j in range(i, i + length)):
                continue
            _, p1, p2 = toks[i]
            if p1 not in pos1 or (pos2 and p2 not in pos2):
                continue
            seq = tuple(t[0] for t in toks[i : i + length])
            for name, f in forms:
                if len(f) == length and seq == f:
                    out[name] += 1
                    used.update(range(i, i + length))
                    break
    return out


def scenes_of(sentences: list[str]) -> list[str]:
    """dialogue / mixed / narration per sentence, carrying the depth of open quotations across sentences."""
    out, depth = [], 0
    for s in sentences:
        if depth > 0 or s.startswith(OPENERS):
            out.append("dialogue")
        elif any(ch in s for ch in "「」『』"):
            out.append("mixed")
        else:
            out.append("narration")
        for ch in s:
            if ch in "「『":
                depth += 1
            elif ch in "」』":
                depth = max(0, depth - 1)
    return out


def analysable(labels: np.ndarray) -> bool:
    counts = Counter(labels.tolist())
    return len(labels) >= 8 and sum(n >= 2 for n in counts.values()) >= 3


def d_and_p(dists: list[Counter], labels: np.ndarray, perms: list[np.ndarray], idx: np.ndarray) -> dict:
    """D (between minus within mean JSD) on the works idx, and its p under the permutations of all 23 labels."""
    D = np.array([[jsd(p, q) if i != j else 0.0 for j, q in enumerate(dists)] for i, p in enumerate(dists)])
    obs = between_minus_within(D, labels[idx])
    null = np.array([between_minus_within(D, p[idx]) for p in perms])
    same = labels[idx][:, None] == labels[idx][None, :]
    iu = np.triu_indices(len(idx), 1)
    return {"D": obs, "between": float(D[iu][~same[iu]].mean()), "within": float(D[iu][same[iu]].mean()),
            "p": float((1 + np.sum(null >= obs)) / (PERMUTATIONS + 1)), "null": null}


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {"sets": 0}
    obs = float(np.mean([r["D"] for r in results]))
    null = np.mean([r["null"] for r in results], axis=0)
    return {"sets": len(results), "mean_D": obs, "p": float((1 + np.sum(null >= obs)) / (PERMUTATIONS + 1))}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    _, w2 = cm.load_conditions(WORK2_COND)
    if cm.sha256(WORK2_COND) != c["work2_conditions_sha256"]:
        raise SystemExit("works/conditions/work2.json differs from the one named in the conditions")
    cm.same_as_head([COND, WORK2_COND, WORK2_RESULT, ROOT / w2["map"]["file"], *(ROOT / p for p in CODE)])
    state = cm.git_state(["checks", "src", "works", "results/work2", "results/calibrate_memolon.json",
                          "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of work 2")
    lemma_tagger = fugashi.GenericTagger(ipadic.MECAB_ARGS)  # the same dictionary, for feature 6
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if lex.sha256 != w2["lexicon"]["sha256"]:
        raise SystemExit("the lexicon differs from that of work 2")
    vmap = calibration_map(cm.check_file(ROOT / w2["map"]["file"], w2["map"]["sha256"], "the calibration"))
    done = {w["work_id"]: w for ws in json.loads(WORK2_RESULT.read_text())["authors"].values() for w in ws}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    A, B = c["partA_variants"], c["partB_scenes"]
    kinds = A["sets"]
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
        if (len(t.points) != ref["sentences"] or len(t.sentences) != len(t.points)
                or sum(p.Z for p in t.points) != ref["matched_tokens"]):
            raise SystemExit(f"{w['work_id']}: the text step gives another result than results/work2")
        variants = {k: {s: Counter() for s in sets} for k, sets in kinds.items()}
        scenes = {sc: {"matched": Counter(), "content": Counter()} for sc in ("narration", "dialogue", "mixed", "all")}
        kinds_of = scenes_of(t.sentences)
        for sent, p, sc in zip(t.sentences, t.points, kinds_of):
            toks = lemmas(lemma_tagger, sent)
            for k, sets in kinds.items():
                for s, spec in sets.items():
                    variants[k][s] += match_set(toks, spec)
            content = Counter(lem for lem, p1, p2 in toks if p2 in japanese_text.POS_ALLOWED.get(p1, ()))
            for key in (sc, "all"):
                scenes[key]["matched"] += Counter(p.matched)
                scenes[key]["content"] += content
        works.append({"author": w["author"], "work_id": w["work_id"], "title": w["title"], "variants": variants,
                      "scenes": scenes, "scene_counts": dict(Counter(kinds_of))})

    authors = sorted({w["author"] for w in works}, key=[w["author"] for w in works].index)
    labels = np.array([authors.index(w["author"]) for w in works])
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(labels) for _ in range(PERMUTATIONS)]

    part_a = {}
    for k, sets in kinds.items():
        done_sets, rows = [], {}
        for s in sets:
            idx = np.array([i for i, w in enumerate(works) if sum(w["variants"][k][s].values()) >= A["min_total"]])
            row = {"included_works": len(idx),
                   "pooled_share": {}, "dominant": {}}
            for a in authors:
                mine = [i for i in idx if works[i]["author"] == a]
                pooled = sum((works[i]["variants"][k][s] for i in mine), Counter())
                n = sum(pooled.values())
                row["pooled_share"][a] = {"n": n, **{v: pooled[v] / n for v in sets[s]["variants"]}} if n else {"n": 0}
                row["dominant"][a] = {}
                for i in mine:
                    cnt = works[i]["variants"][k][s]
                    top = max(cnt.values())
                    tied = [v for v in sets[s]["variants"] if cnt[v] == top]
                    row["dominant"][a][works[i]["work_id"]] = {"variants": tied, "tie": len(tied) > 1}
            if len(idx) and analysable(labels[idx]):
                r = d_and_p([works[i]["variants"][k][s] for i in idx], labels, perms, idx)
                done_sets.append(r)
                row.update({x: r[x] for x in ("D", "between", "within", "p")})
            else:
                row["sparse"] = True
            rows[s] = row
        part_a[k] = {"sets": rows, "aggregate": aggregate(done_sets)}

    part_b = {}
    for sc in ("narration", "dialogue", "all"):  # mixed is reported by its count only
        part_b[sc] = {}
        for voc in ("matched", "content"):
            idx = np.array([i for i, w in enumerate(works) if sum(w["scenes"][sc][voc].values()) >= B["min_tokens"]])
            row = {"included_works": len(idx),
                   "tokens": {w["work_id"]: sum(w["scenes"][sc][voc].values()) for w in works}}
            if len(idx) and analysable(labels[idx]):
                r = d_and_p([works[i]["scenes"][sc][voc] for i in idx], labels, perms, idx)
                row.update({x: r[x] for x in ("D", "between", "within", "p")})
            else:
                row["sparse"] = True
            part_b[sc][voc] = row

    crit = {"C6a_orthography_p": part_a["orthography"]["aggregate"].get("p"),
            "C6b": bool(part_a["lexical"]["aggregate"].get("sets") and part_a["lexical"]["aggregate"]["mean_D"] > 0
                        and part_a["lexical"]["aggregate"]["p"] < 0.05),
            "C6c_reference_p": part_a["reference_to_people"]["aggregate"].get("p"),
            "C6d": all(not part_b["narration"][v].get("sparse") and part_b["narration"][v]["D"] > 0
                       and part_b["narration"][v]["p"] < 0.05 for v in ("matched", "content")),
            "C6e": "by eye, recorded in the decisions"}
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                    "work2_conditions_sha256": cm.sha256(WORK2_COND), "lexicon_sha256": lex.sha256,
                    "dicdir_sha256": tagger.dicdir_sha256, **cm.versions("fugashi", "ipadic", "numpy"),
                    "python": platform.python_version(), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "permutations": PERMUTATIONS, "seed": SEED, "authors": authors,
           "works": [{"author": w["author"], "work_id": w["work_id"], "title": w["title"], "scenes": w["scene_counts"],
                      "variants": {k: {s: dict(cnt) for s, cnt in v.items()} for k, v in w["variants"].items()}}
                     for w in works],
           "partA": part_a, "partB": part_b, "criteria": crit}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"  # built before the file is opened
    with OUT.open("x") as fh:
        fh.write(text)
    print(json.dumps({k: v["aggregate"] for k, v in part_a.items()}, ensure_ascii=False))
    print(json.dumps({sc: {v: {x: r.get(x) for x in ("included_works", "D", "p")} for v, r in d.items()}
                      for sc, d in part_b.items()}, ensure_ascii=False))
    print(json.dumps(crit, ensure_ascii=False))


if __name__ == "__main__":
    main()
