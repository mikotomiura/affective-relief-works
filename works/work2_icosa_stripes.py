"""Where the fine stripes on oblique faces of the icosahedron of work 2 come from: the zoom regions of work2_icosa_v2
drawn with the face grid 1600 and 3200, and the heights themselves seen in the same view.

Usage:  uv run python works/work2_icosa_stripes.py        (refuses to overwrite)
Conditions: works/conditions/work2_icosa_stripes.json, work2_icosa_v2.json and work2_words.json (committed before this
        script was run)
Input:  results/work2_words/words.json, results/work2_icosa_v2/icosa_k4.png
Output: results/work2_icosa_stripes/stripes.png, stripes.json
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
from work2_icosa_v2 import frame, gradients, grid, shade, to_u8  # noqa: E402
from work2_words import icosahedron, taper, z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "work2_icosa_stripes.json"
V2_COND = ROOT / "works" / "conditions" / "work2_icosa_v2.json"
WORDS_COND = ROOT / "works" / "conditions" / "work2_words.json"
WORDS_RESULT = ROOT / "results" / "work2_words" / "words.json"
V2_K4 = ROOT / "results" / "work2_icosa_v2" / "icosa_k4.png"
OUT_DIR = ROOT / "results" / "work2_icosa_stripes"
CODE = ["works/work2_icosa_stripes.py", "works/work2_icosa_v2.py", "works/work2_words.py", "works/_common.py",
        "src/relief.py", "checks/author_texture.py", "src/pipeline.py", "src/japanese_text.py", "src/window.py"]


def region_rays(V: np.ndarray, px: int, k: int, rows: list[int], cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """U, W of the k x k sub-positions of the pixels rows x cols of a view, placed as in view_k() of work2_icosa_v2."""
    ext = 1.08 * np.linalg.norm(V[0])
    centres = np.linspace(-ext, ext, px)
    delta = 2 * ext / (px - 1)
    sub = ((np.arange(k) + 0.5) / k - 0.5) * delta
    u = (centres[:, None] + sub[None, :]).ravel()
    return np.meshgrid(u[cols[0] * k : cols[1] * k], -u[rows[0] * k : rows[1] * k])


def hits(V: np.ndarray, faces: list, edge: float, direction: list[float], U: np.ndarray,
         W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per ray: the face hit (-1 for none) and the domain coordinates (qx, qy) on it, as in shade()."""
    d, right, up, _ = frame(direction)
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
    face = np.where(hit, face, -1)
    qx, qy = np.zeros(U.shape), np.zeros(U.shape)
    for k, (a, b, c) in enumerate(faces):
        m = face == k
        if not m.any():
            continue
        X = origin[m] + t_enter[m][:, None] * (-d)
        e1 = (V[b] - V[a]) / edge
        e2 = (V[c] - V[a]) - np.dot(V[c] - V[a], e1) * e1
        e2 /= np.linalg.norm(e2)
        qx[m], qy[m] = (X - V[a]) @ e1 / edge, (X - V[a]) @ e2 / edge
    return face, qx, qy


def heights_on_faces(words: list[dict], seed: int, n: int, w: dict, eps: float, needed: set[int],
                     n_faces: int) -> list[np.ndarray | None]:
    """z (mm, after the taper) on the n x n grid for the needed faces only; None for the others."""
    tap = taper(n, w["icosahedron"]["taper_width"])
    return [z_mm(words[i]["V"], words[i]["A"], seed, n, w, eps) * tap if i in needed else None
            for i in range(n_faces)]


def blocks(a: np.ndarray, k: int) -> np.ndarray:
    r, c = a.shape[0] // k, a.shape[1] // k
    return a.reshape(r, k, c, k).mean(axis=(1, 3))


def single_face(face: np.ndarray, k: int) -> np.ndarray:
    """Pixels whose k x k rays all hit one face and whose 3 x 3 neighbourhood lies on the same face."""
    r, c = face.shape[0] // k, face.shape[1] // k
    f = face.reshape(r, k, c, k)
    lo, hi = f.min(axis=(1, 3)), f.max(axis=(1, 3))
    one = np.where((lo == hi) & (lo >= 0), lo, -1)
    same = (ndimage.minimum_filter(one, 3, mode="constant", cval=-2) == one) & \
        (ndimage.maximum_filter(one, 3, mode="constant", cval=-2) == one)
    return same & (one >= 0)


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    cm.check_file(V2_COND, c["work2_icosa_v2_conditions_sha256"], V2_COND.name)
    cm.check_file(V2_K4, c["work2_icosa_v2_k4_sha256"], V2_K4.name)
    v2 = json.loads(V2_COND.read_bytes())
    cm.check_file(WORDS_COND, v2["work2_words_conditions_sha256"], WORDS_COND.name)
    cm.same_as_head([COND, V2_COND, WORDS_COND, WORDS_RESULT, V2_K4, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2_words", "results/work2_icosa_v2", "pyproject.toml",
                          "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    w = json.loads(WORDS_COND.read_bytes())
    res_w = json.loads(WORDS_RESULT.read_bytes())
    if v2["work2_words_result_sha256"] != cm.sha256(WORDS_RESULT):
        raise SystemExit("words.json differs from the one named in work2_icosa_v2.json")
    cm.check_generator(w["generator"])
    eps = relief.eps_for(relief.Dimensions(**{k: w["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != res_w["eps"]:
        raise SystemExit(f"eps {eps} differs from that of words.json {res_w['eps']}")
    ic = w["icosahedron"]
    edge, px = ic["edge_mm"], ic["view_px"]
    V, faces = icosahedron(edge)
    if [list(f) for f in faces] != res_w["icosahedron_faces"]:
        raise SystemExit("the faces differ from those of words.json")
    k = cm.need(c, "k", int)
    if k != v2["main_k"]:
        raise SystemExit("k differs from main_k of work2_icosa_v2.json (the control would not hold)")
    v2_k4 = np.asarray(Image.open(V2_K4))
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    trial = w["trial_authors"]
    order = [(a, name) for a in trial for name in ("grazing", "oblique")]
    rows_fig, reported = [], []
    e = v2["zoom"]["enlarge"]
    for i_a, a in enumerate(trial):
        seed = res_w["authors"][a]["seed"]
        words = res_w["authors"][a]["words"]
        if len(words) != len(faces):
            raise SystemExit(f"{a}: {len(words)} words for {len(faces)} faces")
        for name in ("grazing", "oblique"):
            reg = v2["zoom"]["regions"][name]
            if not all(0 <= x0 < x1 <= px for x0, x1 in (reg["rows"], reg["cols"])):
                raise SystemExit(f"the region {name} does not lie inside the view")
            direction = ic["views"][reg["view"]]
            U, W = region_rays(V, px, k, reg["rows"], reg["cols"])
            face, qx, qy = hits(V, faces, edge, direction, U, W)
            needed = {int(f) for f in np.unique(face) if f >= 0}
            out = {}
            for n in (ic["face_grid"], 3200):
                zs = heights_on_faces(words, seed, n, w, eps, needed, len(faces))
                unused = np.full((2, 2), np.nan)  # a face sampled without heights would show as NaN
                grads = [gradients([z], edge)[0] if z is not None else (unused, unused) for z in zs]
                out[f"G{n}"] = blocks(shade(V, faces, grads, edge, n, direction, U, W), k)
                if np.isnan(out[f"G{n}"]).any():
                    raise SystemExit(f"a face without heights was sampled ({a}, {name})")
                if n == 3200:
                    zr = np.full(U.shape, np.nan)
                    for f in needed:
                        m = face == f
                        zr[m] = ndimage.map_coordinates(zs[f], np.vstack([qy[m] * n, qx[m] * n]), order=1,
                                                        mode="nearest")
                    out["Hz"] = zr
                del zs, grads
            # control: G1600 is the crop of icosa_k4.png
            v_col = 0 if reg["view"] == "front" else 1
            r0, c0 = i_a * (px + 16) + reg["rows"][0], v_col * (px + 16) + reg["cols"][0]
            ref = v2_k4[r0 : r0 + reg["rows"][1] - reg["rows"][0], c0 : c0 + reg["cols"][1] - reg["cols"][0]]
            if not np.array_equal(to_u8(out["G1600"]), ref):
                raise SystemExit(f"G1600 does not reproduce icosa_k4.png ({a}, {name})")
            inner = single_face(face, k)
            diff = np.abs(to_u8(out["G3200"]).astype(float) - to_u8(out["G1600"]).astype(float))[inner]
            solid = (face >= 0).reshape(face.shape[0] // k, k, face.shape[1] // k, k).all(axis=(1, 3))
            zpix = np.where(solid, blocks(np.nan_to_num(out["Hz"], nan=0.0), k), 0.0)
            zmin, zmax = float(zpix[solid].min()), float(zpix[solid].max())
            # high-pass: z minus its Gaussian blur over the solid pixels (normalised by the blurred mask)
            wsum = ndimage.gaussian_filter(solid.astype(float), 3)
            low = ndimage.gaussian_filter(zpix, 3) / np.where(wsum > 0, wsum, 1.0)
            hp = zpix - low
            lo, hi = np.percentile(hp[inner], [1, 99])
            if not hi > lo:
                raise SystemExit(f"the high-pass heights have no range ({a}, {name})")
            hmap = np.full(zpix.shape, 0.5)
            hmap[solid] = np.clip((hp[solid] - lo) / (hi - lo), 0.0, 1.0)
            reported.append({"author": a, "region": name, "faces": sorted(needed), "pixels_inner": int(inner.sum()),
                             "G3200_vs_G1600_mean": float(diff.mean()),
                             "G3200_vs_G1600_p99": float(np.percentile(diff, 99)),
                             "highpass_z_sd_um": float(hp[inner].std() * 1000),
                             "highpass_z_abs_p99_um": float(np.percentile(np.abs(hp[inner]), 99) * 1000),
                             "z_min_mm": zmin, "z_max_mm": zmax})
            rows_fig.append([np.kron(to_u8(x), np.ones((e, e), np.uint8)) for x in (out["G1600"], out["G3200"], hmap)])

    tmp = Path(tempfile.mkdtemp(prefix=".work2_icosa_stripes-", dir=OUT_DIR.parent))
    Image.fromarray(grid(rows_fig), mode="L").save(tmp / "stripes.png")
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                    "work2_icosa_v2_conditions_sha256": cm.sha256(V2_COND),
                    "work2_words_conditions_sha256": cm.sha256(WORDS_COND), "icosa_k4_sha256": cm.sha256(V2_K4),
                    "work2_words_result_sha256": cm.sha256(WORDS_RESULT),
                    **cm.versions("numpy", "scipy", "pillow"), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "rows_of_figure": [f"{a} {name}" for a, name in order], "control_G1600_equals_icosa_k4": True,
           "reported": reported, "figures": {"stripes.png": cm.sha256(tmp / "stripes.png")}}
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"  # built before the file is opened
    with (tmp / "stripes.json").open("x") as fh:
        fh.write(text)
    if OUT_DIR.exists():  # rename would replace an empty directory without a sign
        raise SystemExit(f"{OUT_DIR} appeared during the run; the outputs stay in {tmp}")
    tmp.rename(OUT_DIR)
    for r in reported:
        print(r["author"], r["region"], r["faces"], f"mean {r['G3200_vs_G1600_mean']:.2f}",
              f"p99 {r['G3200_vs_G1600_p99']:.1f}", f"hp sd {r['highpass_z_sd_um']:.1f} um",
              f"hp p99 {r['highpass_z_abs_p99_um']:.1f} um", f"z {r['z_min_mm']:.3f}..{r['z_max_mm']:.3f}")


if __name__ == "__main__":
    main()
