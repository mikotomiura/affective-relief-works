"""Relief of one tile from (f, s, gamma), its height scales, and its STL file, written for work 1 (one tile per
sentence).

The decisions (.steering/20261028-while-waiting-2/ decisions 1 and 2, made before any text of the work was computed):
- Field: the pinned generator's procedural_field_separable (4 octaves, normalisation 'ampsum', one seed for every
  tile). It samples x = i/n on [0, 1); the print grid adds the edge x = 1 (and y = 1) so that the tile is exactly L
  wide: N intervals, N + 1 points per side (closed_field).
- Heights: H = s T_gamma(F) and H_eps = s T_{gamma,eps}(F). The STL is made from H_eps (T_gamma has a slope at the
  zero lines that grows with the grid when gamma > 1, so its printed shape has no fixed size).
- Scales: common z = D H / (2 s_max) (z = 0 is the middle of the span D, the same for every tile of a series);
  min-max z = D (H - min H) / (max H - min H) (every tile spans D). The top of the solid is b + D/2 + z (common) or
  b + z (min-max), with b the thickness of the base under the lowest point the scale allows.
- eps is tied to the smallest printable feature e_min by the rule of the paper on the rule: the band around a zero
  line where T_{gamma,eps} departs from T_gamma is about 2 eps L / g wide (g the gradient of F, a linear
  approximation), and it is made e_min wide where F is steepest: eps = e_min G* / (2 L), G* = 88.22697899815242
  (that paper's largest gradient at f = 7 over 20 seeds at N = 4096).
- Grid: points h = e_min / 2 apart (N = L / h intervals).
- STL: binary, a closed solid (top grid, flat bottom, four walls) with outward normals and no T-junction (the
  bottom is a fan from its centre to every point of the boundary, so each edge of a wall meets one edge of the
  bottom). x runs along the columns of the arrays and y along the rows; seen from above, the origin is at the lower
  left (as in the figures, which flip the rows).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from atact.field import procedural_field_separable
from atact.mapping import params
from atact.transform import t_gamma, t_gamma_eps

OCTAVES = 4
NORMALIZE = "ampsum"  # the field is divided by a constant, so a field computed in pieces equals the field computed whole
G_STAR = 88.22697899815242
SCALES = ("common", "minmax")


@dataclass(frozen=True)
class Dimensions:
    L_mm: float = 50.0
    D_mm: float = 1.2
    s_max: float = 0.30
    base_mm: float = 2.0
    e_min_mm: float = 0.4

    def __post_init__(self) -> None:
        for k in ("L_mm", "D_mm", "s_max", "base_mm", "e_min_mm"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{k} = {v!r} is not a positive number")


def eps_for(d: Dimensions, g_star: float = G_STAR) -> float:
    return d.e_min_mm * g_star / (2.0 * d.L_mm)


def grid_intervals(d: Dimensions) -> int:
    """N = L / (e_min / 2); an error unless it is a whole number (to 1e-9), so that the grid spacing is exactly h."""
    x = d.L_mm / (d.e_min_mm / 2.0)
    n = round(x)
    if n < 2 or abs(x - n) > 1e-9:
        raise ValueError(f"L / (e_min / 2) = {x} is not a whole number of intervals")
    return n


def closed_field(f: float, n: int, seed: int) -> np.ndarray:
    """F at x = i/n, y = j/n for i, j = 0..n, as an (n + 1) x (n + 1) array (rows y, columns x). The inner n x n
    block is the generator's own call; the edges x = 1 and y = 1 come from calls with the offset 1 (whose first
    sample is exactly 1.0), and the random phases depend on the seed only."""
    if n < 1:
        raise ValueError(f"{n} intervals")
    out = np.empty((n + 1, n + 1))
    out[:n, :n] = procedural_field_separable(f, n, seed, octaves=OCTAVES, normalize=NORMALIZE)
    out[:n, n] = procedural_field_separable(f, n, seed, octaves=OCTAVES, x0=1.0, normalize=NORMALIZE)[:, 0]
    out[n, :n] = procedural_field_separable(f, n, seed, octaves=OCTAVES, y0=1.0, normalize=NORMALIZE)[0, :]
    out[n, n] = procedural_field_separable(f, 1, seed, octaves=OCTAVES, x0=1.0, y0=1.0, normalize=NORMALIZE)[0, 0]
    return out


@dataclass(frozen=True)
class Tile:
    V: float
    A: float
    f: float
    s: float
    gamma: float
    H: np.ndarray
    H_eps: np.ndarray


def tile(V: float, A: float, F_of_f, eps: float) -> Tile:
    """Heights of one tile; F_of_f(f) gives the field on the wanted grid (so that figures and STL share the code)."""
    f, s, g = (float(x) for x in params(V, A))
    F = F_of_f(f)
    return Tile(V, A, f, s, g, s * t_gamma(F, g), s * t_gamma_eps(F, g, eps))


def scaled(H: np.ndarray, scale: str, d: Dimensions) -> np.ndarray:
    """z in mm: common (0 is the middle of D) or min-max (0 to D)."""
    if scale == "common":
        return d.D_mm * H / (2.0 * d.s_max)
    if scale == "minmax":
        span = H.max() - H.min()
        if not span > 0:
            raise ValueError("a flat tile has no min-max scale")
        return d.D_mm * (H - H.min()) / span
    raise ValueError(f"unknown scale {scale!r}")


def top_surface(H: np.ndarray, scale: str, d: Dimensions) -> np.ndarray:
    """Height of the top of the solid above its bottom, in mm."""
    z = scaled(H, scale, d)
    return d.base_mm + (d.D_mm / 2.0 + z if scale == "common" else z)


def grey(H: np.ndarray, scale: str, d: Dimensions) -> np.ndarray:
    """Grey level of the figures: common z / D + 1/2, min-max z / D (as in the text example of the paper)."""
    z = scaled(H, scale, d)
    return z / d.D_mm + (0.5 if scale == "common" else 0.0)


# ---------------------------------------------------------------- solid and STL


def solid(top: np.ndarray, L: float) -> np.ndarray:
    """Triangles (m, 3, 3) of the closed solid under the top surface top[j, i] at x = L i/n, y = L j/n."""
    if not (math.isfinite(L) and L > 0):
        raise ValueError(f"side {L}")
    if top.ndim != 2 or top.shape[0] != top.shape[1] or top.shape[0] < 2:
        raise ValueError(f"top of shape {top.shape}")
    if not np.all(np.isfinite(top)) or not np.all(top > 0):
        raise ValueError("the top must be finite and above the bottom")
    n = top.shape[0] - 1
    c = L * np.arange(n + 1) / n
    X, Y = np.meshgrid(c, c)  # X[j, i] = c[i], Y[j, i] = c[j]
    P = np.stack([X, Y, top], axis=-1)
    v00, v10, v11, v01 = P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]
    tris = [np.stack([v00, v10, v11], axis=-2).reshape(-1, 3, 3),  # counterclockwise seen from above: normal up
            np.stack([v00, v11, v01], axis=-2).reshape(-1, 3, 3)]
    # boundary, counterclockwise seen from above: south (y = 0), east, north, west
    ring = np.concatenate([P[0, :-1], P[:-1, -1], P[-1, :0:-1], P[:0:-1, 0]])
    nxt = np.roll(ring, -1, axis=0)
    a_b, b_b = ring.copy(), nxt.copy()
    a_b[:, 2] = 0.0
    b_b[:, 2] = 0.0
    tris.append(np.stack([a_b, b_b, nxt], axis=1))  # walls: outward
    tris.append(np.stack([a_b, nxt, ring], axis=1))
    centre = np.broadcast_to(np.array([L / 2.0, L / 2.0, 0.0]), a_b.shape)
    tris.append(np.stack([centre, b_b, a_b], axis=1))  # bottom: normal down
    return np.concatenate(tris)


def normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    if not np.all(ln > 0):
        raise ValueError("degenerate triangle")
    return n / ln


_FACET = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def write_stl(path: Path, tris: np.ndarray, header: str) -> None:
    """Binary STL (little endian), refusing to overwrite. Coordinates in mm, as float32."""
    h = header.encode("ascii")
    if len(h) > 80 or h.lower().startswith(b"solid"):  # a header starting with 'solid' is read as ASCII STL by some
        raise ValueError("the header must be ASCII, at most 80 bytes, and not start with 'solid'")
    rec = np.zeros(len(tris), dtype=_FACET)
    t32 = tris.astype("<f4")
    rec["v"] = t32
    rec["n"] = normals(t32.astype(np.float64)).astype("<f4")
    with Path(path).open("xb") as fh:
        fh.write(h.ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        fh.write(rec.tobytes())


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Normals (m, 3) and triangles (m, 3, 3) of a binary STL file; an error if the size does not match the count."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError("too short for a binary STL")
    (m,) = struct.unpack("<I", raw[80:84])
    if len(raw) != 84 + 50 * m:
        raise ValueError(f"{len(raw)} bytes for {m} facets")
    rec = np.frombuffer(raw, dtype=_FACET, offset=84, count=m)
    return rec["n"].astype(np.float64), rec["v"].astype(np.float64)


def signed_volume(tris: np.ndarray) -> float:
    """Volume enclosed by a closed triangle mesh (positive when the normals point outwards)."""
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)


def edge_check(tris: np.ndarray) -> dict:
    """Each undirected edge must belong to exactly two triangles that run along it in opposite directions (closed,
    consistently oriented, no T-junction). Vertices are matched by their exact coordinates."""
    v = tris.reshape(-1, 3)
    _, idx = np.unique(v, axis=0, return_inverse=True)
    idx = idx.reshape(-1, 3)
    e = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
    lo, hi = e.min(axis=1), e.max(axis=1)
    fwd = (e[:, 0] < e[:, 1]).astype(np.int64)
    key = lo.astype(np.int64) * (int(idx.max()) + 1) + hi
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    fsum = np.bincount(inv, weights=fwd, minlength=len(uniq))
    return {"edges": int(len(uniq)), "not_two": int(np.sum(cnt != 2)), "same_direction": int(np.sum((cnt == 2) & (fsum != 1)))}
