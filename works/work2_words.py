"""Work 2 redesigned: the distinctive words of every author (in at least two of its works), and the sketches of the
trial authors as tiles and as an icosahedron (images only).

Usage:  uv run python works/work2_words.py        (refuses to overwrite)
Conditions: works/conditions/work2_words.json, and the inputs of works/conditions/work2.json (committed before this
        script was run)
Input:  the 23 Aozora Bunko zips, MEmoLon, the map B and the MeCab dictionary of work 2 (external/, results/)
Output: results/work2_words/words.json, tiles.png, tiles_common_seed.png, icosa.png

Steps: the Japanese text step of work 2 per work (checked against results/work2), the matched words counted per
work, the selection of the conditions (the statistic of check 5 with a filter on works and on the share of one work),
then per word (V, A) -> (f, s, gamma) and the height H_eps on the field with warp of the author's seed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import relief  # noqa: E402
from atact.field import RenderConfig, procedural_field  # noqa: E402
from atact.mapping import params  # noqa: E402
from atact.transform import t_gamma_eps  # noqa: E402
from author_texture import log_odds  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402

COND = ROOT / "works" / "conditions" / "work2_words.json"
WORK2_COND = ROOT / "works" / "conditions" / "work2.json"
WORK2_RESULT = ROOT / "results" / "work2" / "work2.json"
OUT_DIR = ROOT / "results" / "work2_words"
CODE = ["works/work2_words.py", "works/_common.py", "checks/author_texture.py", "src/pipeline.py",
        "src/japanese_text.py", "src/window.py", "src/relief.py"]
PRIOR_TOTAL = 1000.0  # as check 5


# ---------------------------------------------------------------- selection


def select(words: dict[str, Counter], author_of: dict[str, str], tokens: dict[str, int], min_count: int,
           min_works: int, max_share: float, n_words: int) -> dict[str, list[dict]]:
    """Per author, the candidates in z order (at most n_words). words: per work id, the counts of its matched words."""
    authors = sorted(set(author_of.values()), key=list(author_of.values()).index)
    by_author = {a: sum((words[w] for w in words if author_of[w] == a), Counter()) for a in authors}
    total = sum(by_author.values(), Counter())
    n_total = sum(total.values())
    prior = {w: PRIOR_TOTAL * n / n_total for w, n in total.items()}
    out = {}
    for a in authors:
        own = [w for w in words if author_of[w] == a]
        rest = sum((by_author[b] for b in authors if b != a), Counter())
        z = log_odds(by_author[a], rest, prior)
        cands = []
        for word, zw in z.items():
            if not zw > 0:
                continue
            counts = {w: words[w][word] for w in own}
            if sum(c >= min_count for c in counts.values()) < min_works:
                continue
            rates = [counts[w] / tokens[w] for w in own]
            share = max(rates) / sum(rates)
            if share > max_share + 1e-12:
                continue
            cands.append({"word": word, "z": zw, "share": share, "counts": counts})
        cands.sort(key=lambda d: (-d["z"], d["word"]))
        out[a] = {"candidates": len(cands), "words": cands[:n_words]}
    return out


# ---------------------------------------------------------------- heights


def field_config(c: dict, n: int) -> RenderConfig:
    f = c["field"]
    return RenderConfig(n=n, octaves=f["octaves"], warp=f["warp"], normalize=f["normalize"], freq_unit=f["freq_unit"],
                        octave_form=f["octave_form"])


def z_mm(V: float, A: float, seed: int, n: int, c: dict, eps: float) -> np.ndarray:
    """Common-scale height in mm on the n x n grid of the domain (rows y, columns x)."""
    f, s, g = (float(x) for x in params(V, A))
    F = procedural_field(f, field_config(c, n), seed)
    r = c["relief"]
    return r["D_mm"] * s * t_gamma_eps(F, g, eps) / (2.0 * r["s_max"])


# ---------------------------------------------------------------- tiles


def tile_figure(blocks: list[tuple[str, list[np.ndarray], list[str]]], c: dict, font_path: str) -> Image.Image:
    t = c["tiles"]
    px, gap, lab, cols, rows = t["tile_px"], t["gap_px"], t["label_px"], t["columns"], t["rows"]
    bw = cols * px + (cols - 1) * gap
    head = 40
    W = len(blocks) * bw + (len(blocks) - 1) * t["block_gap_px"]
    H = head + rows * (px + lab) + (rows - 1) * gap
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 18)
    D = c["relief"]["D_mm"]
    for b, (title, tiles, labels) in enumerate(blocks):
        x0 = b * (bw + t["block_gap_px"])
        draw.text((x0, 8), title, fill=0, font=font)
        for k, (z, word) in enumerate(zip(tiles, labels)):
            r, col = divmod(k, cols)
            x, y = x0 + col * (px + gap), head + r * (px + lab + gap)
            grey = np.clip(z / D + 0.5, 0.0, 1.0)
            img.paste(Image.fromarray(np.round(np.flipud(grey) * 255).astype(np.uint8), mode="L"), (x, y))
            draw.text((x, y + px + 4), f"{k + 1} {word}", fill=0, font=font)
    return img


# ---------------------------------------------------------------- icosahedron


def icosahedron(edge: float) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Vertices (12 x 3) and the 20 faces as vertex triples (A, B, C) in the order and frame of the conditions."""
    phi = (1 + math.sqrt(5)) / 2
    v = []
    for a, b in itertools.product((-1, 1), repeat=2):
        v += [(0, a, b * phi), (a, b * phi, 0), (a * phi, 0, b)]
    V = np.array(v, float) * edge / 2
    faces = [t for t in itertools.combinations(range(12), 3)
             if all(abs(np.linalg.norm(V[i] - V[j]) - edge) < 1e-9 for i, j in itertools.combinations(t, 2))]

    def az(p: np.ndarray) -> float:
        return math.atan2(p[1], p[0]) % (2 * math.pi)

    framed = []
    for t in faces:
        apex = max(t, key=lambda i: (round(V[i][2], 9), -az(V[i])))
        a, b = (i for i in t if i != apex)
        centre = V[list(t)].mean(axis=0)
        if np.dot(np.cross(V[b] - V[a], V[apex] - V[a]), centre) < 0:
            a, b = b, a
        framed.append((a, b, apex))
    framed.sort(key=lambda t: (-round(V[list(t)].mean(axis=0)[2], 9), az(V[list(t)].mean(axis=0))))
    return V, framed


def taper(n: int, width: float) -> np.ndarray:
    """smoothstep(d / (width r)) on the n x n grid of the domain; 0 outside the triangle (0,0), (1,0), (1/2, sqrt3/2)."""
    y, x = np.mgrid[0:n, 0:n] / n
    s3 = math.sqrt(3)
    d = np.minimum.reduce([y, (s3 * x - y) / 2, (s3 * (1 - x) - y) / 2])  # distances to the three edges
    t = np.clip(d / (width * s3 / 6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def view(V: np.ndarray, faces: list, zs: list[np.ndarray], edge: float, direction: list[float], px: int) -> np.ndarray:
    """Grey image (rows top to bottom) of the solid seen along -direction, bump-shaded with the heights zs per face."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    right = np.cross([0.0, 0.0, 1.0], d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    light = 0.45 * d + 0.65 * up - 0.60 * right
    light /= np.linalg.norm(light)
    R = np.linalg.norm(V[0])
    ext = 1.08 * R
    u = np.linspace(-ext, ext, px)
    U, W = np.meshgrid(u, -u)  # rows from the top
    origin = U[..., None] * right + W[..., None] * up + 3 * R * d
    normals = np.array([np.cross(V[b] - V[a], V[c] - V[a]) for a, b, c in faces])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    h = np.array([np.dot(nm, V[a]) for nm, (a, _, _) in zip(normals, faces)])
    denom = normals @ (-d)
    t = (h[None, None, :] - origin @ normals.T) / denom
    t_enter = np.where(denom < 0, t, -np.inf).max(axis=-1)
    t_exit = np.where(denom > 0, t, np.inf).min(axis=-1)
    face = np.where(denom < 0, t, -np.inf).argmax(axis=-1)
    hit = t_enter <= t_exit
    img = np.ones((px, px))
    n_grid = zs[0].shape[0]
    for k, (a, b, c) in enumerate(faces):
        m = hit & (face == k)
        if not m.any():
            continue
        X = origin[m] + t_enter[m][:, None] * (-d)
        e1 = (V[b] - V[a]) / edge
        e2 = (V[c] - V[a]) - np.dot(V[c] - V[a], e1) * e1
        e2 /= np.linalg.norm(e2)
        qx, qy = (X - V[a]) @ e1 / edge, (X - V[a]) @ e2 / edge
        gy, gx = np.gradient(zs[k], edge / n_grid)  # mm per mm along y (rows) and x (columns)
        coords = np.vstack([qy * n_grid, qx * n_grid])
        sx = ndimage.map_coordinates(gx, coords, order=1, mode="nearest")
        sy = ndimage.map_coordinates(gy, coords, order=1, mode="nearest")
        nrm = normals[k][None, :] - sx[:, None] * e1 - sy[:, None] * e2
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        img[m] = 0.15 + 0.85 * (1.0 + nrm @ light) / 2.0  # half-Lambert: the relief stays visible on faces turned away
    return img


# ---------------------------------------------------------------- run


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    _, w2 = cm.load_conditions(WORK2_COND)
    if cm.sha256(WORK2_COND) != c["work2_conditions_sha256"]:
        raise SystemExit("works/conditions/work2.json differs from the one named in the conditions")
    cm.same_as_head([COND, WORK2_COND, WORK2_RESULT, ROOT / w2["map"]["file"], *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2", "results/calibrate_memolon.json",
                          "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_generator(c["generator"])
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of work 2")
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if lex.sha256 != w2["lexicon"]["sha256"]:
        raise SystemExit("the lexicon differs from that of work 2")
    vmap = calibration_map(cm.check_file(ROOT / w2["map"]["file"], w2["map"]["sha256"], "the calibration"))
    authors = [a["author"] for a in c["authors"]]
    if authors != sorted({w["author"] for w in w2["works"]}, key=[w["author"] for w in w2["works"]].index):
        raise SystemExit("the authors of the conditions differ from those of work 2 (or their order)")
    seed_of = {a["author"]: a["seed"] for a in c["authors"]}
    slug_of = {a["author"]: a["slug"] for a in c["authors"]}
    font = c["tiles"]["font_file"]
    done = {w["work_id"]: w for ws in json.loads(WORK2_RESULT.read_text())["authors"].values() for w in ws}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    words, author_of, tokens, titles = {}, {}, {}, {}
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
            raise SystemExit(f"{w['work_id']}: the text step gives another result than results/work2")
        words[w["work_id"]] = Counter(x for p in t.points for x in p.matched)
        author_of[w["work_id"]], titles[w["work_id"]] = w["author"], w["title"]
        tokens[w["work_id"]] = sum(words[w["work_id"]].values())

    sel = c["selection"]
    chosen = select(words, author_of, tokens, sel["min_count"], sel["min_works"], sel["max_share"], sel["n_words"])
    short = {a: r["candidates"] for a, r in chosen.items() if r["candidates"] < sel["n_words"]}
    if short:
        raise SystemExit(f"fewer than {sel['n_words']} candidates: {short}")
    eps = relief.eps_for(relief.Dimensions(**{k: c["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                        "e_min_mm")}))

    def va(word: str) -> tuple[float, float]:
        return vmap(*lex.entries[word])

    per_author = {}
    for a in authors:
        rows = []
        for d in chosen[a]["words"]:
            V, A = va(d["word"])
            f, s, g = (float(x) for x in params(V, A))
            rows.append({**d, "counts": {titles[w]: n for w, n in d["counts"].items()}, "V": V, "A": A, "f": f,
                         "s": s, "gamma": g})
        spread = {k: float(np.std([r[k] for r in rows])) for k in ("V", "A", "f", "s", "gamma")}
        rho = float(stats.spearmanr([r["z"] for r in rows], [r["V"] for r in rows]).statistic)
        per_author[a] = {"slug": slug_of[a], "seed": seed_of[a], "candidates": chosen[a]["candidates"], "words": rows,
                         "spread": spread, "spearman_z_V": rho}

    # sensitivity (reported only)
    base = {a: [d["word"] for d in chosen[a]["words"]] for a in authors}
    thresholds = []
    th = c["sensitivity"]["thresholds"]
    for mc, ms in itertools.product(th["min_count"], th["max_share"]):
        r = select(words, author_of, tokens, mc, sel["min_works"], ms, sel["n_words"])
        thresholds.append({"min_count": mc, "max_share": ms,
                           "per_author": {a: {"candidates": r[a]["candidates"],
                                              "overlap": len(set(base[a]) & {d["word"] for d in r[a]["words"]})}
                                          for a in authors}})
    loo = []
    for w in words:
        keep = [x for x in words if x != w]
        r = select({x: words[x] for x in keep}, {x: author_of[x] for x in keep}, {x: tokens[x] for x in keep},
                   sel["min_count"], sel["min_works"], sel["max_share"], sel["n_words"])
        a = author_of[w]
        loo.append({"work": titles[w], "author": a, "candidates": r[a]["candidates"],
                    "retained": len(set(base[a]) & {d["word"] for d in r[a]["words"]}) / sel["n_words"]})

    # sketches of the trial authors, written into a temporary directory that becomes OUT_DIR only at the end
    tmp = Path(tempfile.mkdtemp(prefix=".work2_words-", dir=OUT_DIR.parent))
    tp, k = c["tiles"]["tile_px"], c["tiles"]["supersample"]
    trial = c["trial_authors"]

    def tiles_of(a: str, seed: int) -> list[np.ndarray]:
        # rendered k times finer and averaged over k x k blocks (the warped field exceeds the Nyquist limit at tp)
        return [z_mm(r["V"], r["A"], seed, tp * k, c, eps).reshape(tp, k, tp, k).mean(axis=(1, 3))
                for r in per_author[a]["words"]]

    own = [(f"{slug_of[a]} (seed {seed_of[a]})", tiles_of(a, seed_of[a]), [r["word"] for r in per_author[a]["words"]])
           for a in trial]
    common = [(f"{slug_of[a]} (common seed {c['common_seed']})", tiles_of(a, c["common_seed"]),
               [r["word"] for r in per_author[a]["words"]]) for a in trial]
    figs = {}
    for name, blocks in (("tiles.png", own), ("tiles_common_seed.png", common)):
        tile_figure(blocks, c, font).save(tmp / name)
        figs[name] = cm.sha256(tmp / name)
    ic = c["icosahedron"]
    V, faces = icosahedron(ic["edge_mm"])
    n = ic["face_grid"]
    tap, px = taper(n, ic["taper_width"]), ic["view_px"]
    views = []
    for a in trial:
        zs = [z_mm(r["V"], r["A"], seed_of[a], n, c, eps) * tap for r in per_author[a]["words"]]
        views.append([view(V, faces, zs, ic["edge_mm"], ic["views"][v], px) for v in ("front", "back")])
        del zs
    step = px + 16
    canvas = np.ones((len(views) * step - 16, 2 * step - 16))
    for i, row in enumerate(views):
        for j, im in enumerate(row):
            canvas[i * step : i * step + px, j * step : j * step + px] = im
    Image.fromarray(np.round(np.clip(canvas, 0, 1) * 255).astype(np.uint8), mode="L").save(tmp / "icosa.png")
    figs["icosa.png"] = cm.sha256(tmp / "icosa.png")

    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                    "work2_conditions_sha256": cm.sha256(WORK2_COND), "lexicon_sha256": lex.sha256,
                    "dicdir_sha256": tagger.dicdir_sha256, "font_sha256": cm.sha256(Path(font)),
                    **cm.versions("fugashi", "ipadic", "numpy", "scipy", "pillow"), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "eps": eps, "authors": per_author, "sensitivity": {"thresholds": thresholds, "leave_one_work_out": loo},
           "icosahedron_faces": [[int(i) for i in f] for f in faces], "figures": figs}
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"  # built before the file is opened
    with (tmp / "words.json").open("x") as fh:
        fh.write(text)
    tmp.rename(OUT_DIR)  # fails if OUT_DIR appeared meanwhile
    for a in authors:
        p = per_author[a]
        print(a, p["candidates"], " ".join(r["word"] for r in p["words"]))


if __name__ == "__main__":
    main()
