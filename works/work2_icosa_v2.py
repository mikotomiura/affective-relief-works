"""The icosahedron of work 2 drawn again: the same words, seeds and relief as results/work2_words/, shaded at k x k
sub-positions per pixel and averaged (the views of work2_words.py take one ray per pixel and alias on oblique faces).

Usage:  uv run python works/work2_icosa_v2.py        (refuses to overwrite)
Conditions: works/conditions/work2_icosa_v2.json, and works/conditions/work2_words.json (committed before this script
        was run)
Input:  results/work2_words/words.json
Output: results/work2_icosa_v2/icosa_k4.png, icosa_k1_k4.png, zoom.png, icosa_v2.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
from work2_words import icosahedron, taper, z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "work2_icosa_v2.json"
WORDS_COND = ROOT / "works" / "conditions" / "work2_words.json"
WORDS_RESULT = ROOT / "results" / "work2_words" / "words.json"
OLD_IMAGE = ROOT / "results" / "work2_words" / "icosa.png"
OUT_DIR = ROOT / "results" / "work2_icosa_v2"
CODE = ["works/work2_icosa_v2.py", "works/work2_words.py", "works/_common.py", "src/relief.py",
        "checks/author_texture.py", "src/pipeline.py", "src/japanese_text.py", "src/window.py"]  # all that is imported


def frame(direction: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """d, right, up and light of a view, as in view() of work2_words.py."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    right = np.cross([0.0, 0.0, 1.0], d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    light = 0.45 * d + 0.65 * up - 0.60 * right
    light /= np.linalg.norm(light)
    return d, right, up, light


def shade(V: np.ndarray, faces: list, grads: list[tuple[np.ndarray, np.ndarray]], edge: float, n_grid: int,
          direction: list[float], U: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Shade of the rays through the points (U right, W up) of the view plane; 1.0 where a ray misses the solid. The
    body of view() of work2_words.py, with the gradients of the faces computed once by the caller."""
    d, right, up, light = frame(direction)
    R = np.linalg.norm(V[0])
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
    img = np.ones(U.shape)
    for k, (a, b, c) in enumerate(faces):
        m = hit & (face == k)
        if not m.any():
            continue
        X = origin[m] + t_enter[m][:, None] * (-d)
        e1 = (V[b] - V[a]) / edge
        e2 = (V[c] - V[a]) - np.dot(V[c] - V[a], e1) * e1
        e2 /= np.linalg.norm(e2)
        qx, qy = (X - V[a]) @ e1 / edge, (X - V[a]) @ e2 / edge
        gy, gx = grads[k]
        coords = np.vstack([qy * n_grid, qx * n_grid])
        sx = ndimage.map_coordinates(gx, coords, order=1, mode="nearest")
        sy = ndimage.map_coordinates(gy, coords, order=1, mode="nearest")
        nrm = normals[k][None, :] - sx[:, None] * e1 - sy[:, None] * e2
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        img[m] = 0.15 + 0.85 * (1.0 + nrm @ light) / 2.0
    return img


def gradients(zs: list[np.ndarray], edge: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """(gy, gx) of every face in mm per mm, as view() of work2_words.py."""
    return [tuple(np.gradient(z, edge / z.shape[0])) for z in zs]


def view_k(V: np.ndarray, faces: list, grads: list[tuple[np.ndarray, np.ndarray]], edge: float,
           direction: list[float], px: int, k: int, rows_per_chunk: int) -> np.ndarray:
    """The view of view() of work2_words.py, each pixel the mean of k x k sub-positions (k = 1: the pixel centres)."""
    n_grid = grads[0][0].shape[0]
    ext = 1.08 * np.linalg.norm(V[0])
    centres = np.linspace(-ext, ext, px)
    delta = 2 * ext / (px - 1)
    sub = ((np.arange(k) + 0.5) / k - 0.5) * delta
    u = (centres[:, None] + sub[None, :]).ravel()  # px * k positions, the k of pixel i together
    out = np.empty((px, px))
    for r0 in range(0, px, rows_per_chunk):
        r1 = min(px, r0 + rows_per_chunk)
        U, W = np.meshgrid(u, -u[r0 * k : r1 * k])  # rows from the top
        img = shade(V, faces, grads, edge, n_grid, direction, U, W)
        out[r0:r1] = img.reshape(r1 - r0, k, px, k).mean(axis=(1, 3))
    return out


def to_u8(img: np.ndarray) -> np.ndarray:
    return np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)


def grid(images: list[list[np.ndarray]], gap: int = 16) -> np.ndarray:
    """Rows of equally sized uint8 images on white, gap px apart."""
    h, w = images[0][0].shape
    cols = max(len(r) for r in images)
    canvas = np.full((len(images) * (h + gap) - gap, cols * (w + gap) - gap), 255, np.uint8)
    for i, row in enumerate(images):
        for j, im in enumerate(row):
            canvas[i * (h + gap) : i * (h + gap) + h, j * (w + gap) : j * (w + gap) + w] = im
    return canvas


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    for key, p in (("work2_words_conditions_sha256", WORDS_COND), ("work2_words_result_sha256", WORDS_RESULT)):
        cm.check_file(p, c[key], p.name)
    cm.same_as_head([COND, WORDS_COND, WORDS_RESULT, OLD_IMAGE, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2_words", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    w = json.loads(WORDS_COND.read_bytes())
    res_w = json.loads(WORDS_RESULT.read_bytes())
    cm.check_generator(w["generator"])
    eps = relief.eps_for(relief.Dimensions(**{k: w["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != res_w["eps"]:
        raise SystemExit(f"eps {eps} differs from that of words.json {res_w['eps']}")
    ic = w["icosahedron"]
    V, faces = icosahedron(ic["edge_mm"])
    if [list(f) for f in faces] != res_w["icosahedron_faces"]:
        raise SystemExit("the faces differ from those of words.json")
    if cm.sha256(OLD_IMAGE) != res_w["figures"]["icosa.png"]:
        raise SystemExit("icosa.png differs from the one named in words.json")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    n, px, chunk = ic["face_grid"], ic["view_px"], cm.need(c, "rows_per_chunk", int)
    ks, kmain = c["k"], cm.need(c, "main_k", int)
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks) or ks[0] != 1 \
            or kmain not in ks or chunk < 1:
        raise SystemExit("k must be positive integers starting with 1 and containing main_k; rows_per_chunk >= 1")
    tap = taper(n, ic["taper_width"])
    trial = w["trial_authors"]
    views: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for a in trial:
        seed = res_w["authors"][a]["seed"]
        grads = gradients([z_mm(r["V"], r["A"], seed, n, w, eps) * tap for r in res_w["authors"][a]["words"]],
                          ic["edge_mm"])
        views[a] = {}
        for v in ("front", "back"):
            views[a][v] = {}
            for k in ks:
                views[a][v][k] = view_k(V, faces, grads, ic["edge_mm"], ic["views"][v], px, k, chunk)
                if k == 1 and a == trial[0] and v == "front":  # fail early: the first view against the old image
                    if not np.array_equal(to_u8(views[a][v][1]), np.asarray(Image.open(OLD_IMAGE))[:px, :px]):
                        raise SystemExit("k = 1 does not reproduce the first view of results/work2_words/icosa.png")
        del grads

    # control: k = 1 is the old image, pixel for pixel
    old = grid([[to_u8(views[a][v][1]) for v in ("front", "back")] for a in trial])
    if not np.array_equal(old, np.asarray(Image.open(OLD_IMAGE))):
        raise SystemExit("k = 1 does not reproduce results/work2_words/icosa.png")

    convergence = []
    for a in trial:
        for v in ("front", "back"):
            ref = to_u8(views[a][v][kmain]).astype(float)
            regions = {"solid": views[a][v][kmain] < 1.0,  # with the outline
                       "interior": ndimage.maximum_filter(views[a][v][1], 3, mode="constant", cval=1.0) < 1.0}
            for k in ks:
                if k == kmain:
                    continue
                for name, m in regions.items():
                    diff = np.abs(to_u8(views[a][v][k]).astype(float) - ref)[m]
                    convergence.append({"author": a, "view": v, "k": k, "vs_k": kmain, "pixels": name,
                                        "n": int(m.sum()), "mean": float(diff.mean()),
                                        "p99": float(np.percentile(diff, 99))})

    tmp = Path(tempfile.mkdtemp(prefix=".work2_icosa_v2-", dir=OUT_DIR.parent))
    figs = {f"icosa_k{kmain}.png": grid([[to_u8(views[a][v][kmain]) for v in ("front", "back")] for a in trial]),
            f"icosa_k1_k{kmain}.png": grid([[to_u8(views[a]["front"][k]) for k in (1, kmain)] for a in trial])}
    z = c["zoom"]
    e = z["enlarge"]
    zoom_rows = []
    for a in trial:
        row = []
        for name in ("grazing", "oblique"):
            reg = z["regions"][name]
            (r0, r1), (c0, c1) = reg["rows"], reg["cols"]
            for k in (1, kmain):
                crop = to_u8(views[a][reg["view"]][k])[r0:r1, c0:c1]
                row.append(np.kron(crop, np.ones((e, e), np.uint8)))
        zoom_rows.append(row)
    figs["zoom.png"] = grid(zoom_rows)
    hashes = {}
    for name, arr in figs.items():
        Image.fromarray(arr, mode="L").save(tmp / name)
        hashes[name] = cm.sha256(tmp / name)

    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                    "work2_words_conditions_sha256": cm.sha256(WORDS_COND),
                    "work2_words_result_sha256": cm.sha256(WORDS_RESULT),
                    **cm.versions("numpy", "scipy", "pillow"), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "eps": eps, "control_k1_equals_icosa_png": True, "convergence": convergence, "figures": hashes}
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"  # built before the file is opened
    with (tmp / "icosa_v2.json").open("x") as fh:
        fh.write(text)
    if OUT_DIR.exists():  # rename would replace an empty directory without a sign
        raise SystemExit(f"{OUT_DIR} appeared during the run; the outputs stay in {tmp}")
    tmp.rename(OUT_DIR)
    for r in convergence:
        print(r["author"], r["view"], f"k={r['k']} vs {r['vs_k']}", r["pixels"], f"mean {r['mean']:.2f}",
              f"p99 {r['p99']:.1f}")


if __name__ == "__main__":
    main()
