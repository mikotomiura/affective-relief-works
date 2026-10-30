"""Images of the README: the two works side by side, reduced for the page, and an animation of how one sentence of work 1 becomes a
tile. The animation steps through the first sentences of work 1 as recorded in results/work1/work1.json (their mean V
and A, f, s and gamma); each frame shows the tile of that sentence drawn as in Fig. 4 of the paper (gray height under
the common scale, and lit from the upper left). It is an illustration: no number of the paper comes from it.

Usage:  uv run python tools/readme_media.py      (writes docs/media/; overwrites)
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import relief  # noqa: E402
import work1  # noqa: E402
import work1_drawing as w1d  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.mapping import params  # noqa: E402

OUT = ROOT / "docs" / "media"
FONT = "/System/Library/Fonts/Helvetica.ttc"
N_SENTENCES = 24
TILE = 200
R = 0.7  # the half width of the plane drawn (all sentences of work 1 lie within it; checked in main)
INK, MUTED, FAINT, ACCENT, PAPER = (34, 34, 34), (110, 110, 110), (205, 205, 205), (214, 110, 40), (255, 255, 255)


def banner(left: Path, right: Path, dst: Path, height: int = 640, gap: int = 40) -> None:
    """The two works side by side at one height, on white."""
    a, b = (Image.open(x).convert("RGB") for x in (left, right))
    a, b = (im.resize((round(im.width * height / im.height), height), Image.LANCZOS) for im in (a, b))
    out = Image.new("RGB", (a.width + gap + b.width, height), PAPER)
    out.paste(a, (0, 0))
    out.paste(b, (a.width + gap, 0))
    out.save(dst, optimize=True)


def tiles_of(p: dict, d: relief.Dimensions, eps: float, seed: int, el: float, az: float) -> tuple[Image.Image, ...]:
    def heights(n: int) -> np.ndarray:
        return relief.tile(p["V"], p["A"], lambda f: procedural_field_separable(
            f, n, seed, octaves=relief.OCTAVES, normalize=relief.NORMALIZE), eps).H_eps

    gray = relief.grey(heights(TILE), "common", d)
    k = 2
    lit = w1d.block_mean(w1d.lambert(relief.scaled(heights(TILE * k), "common", d), d.L_mm, az, el), k)
    return tuple(Image.fromarray(w1d.u8(g)).convert("RGB") for g in (gray, lit))


def frame(i: int, sentences: list[dict], n_all: int, tiles: tuple[Image.Image, ...], fonts: dict) -> Image.Image:
    W, H = 1000, 360
    im = Image.new("RGB", (W, H), PAPER)
    dr = ImageDraw.Draw(im)
    p = sentences[i]
    dr.text((24, 18), f"Work 1 (\"A Pair of Silk Stockings\"), sentence {i + 1} of {n_all}", font=fonts["small"], fill=MUTED)
    lines = textwrap.wrap(p["text"], 110)[:2]
    if len(textwrap.wrap(p["text"], 110)) > 2:
        lines[-1] = lines[-1].rstrip(".,;") + " ..."
    for j, ln in enumerate(lines):
        dr.text((24, 40 + 20 * j), ln, font=fonts["text"], fill=INK)
    # the plane of valence and arousal
    x0, y0, side = 40, 104, 200
    dr.rectangle((x0, y0, x0 + side, y0 + side), outline=FAINT)
    dr.line((x0 + side / 2, y0, x0 + side / 2, y0 + side), fill=FAINT)
    dr.line((x0, y0 + side / 2, x0 + side, y0 + side / 2), fill=FAINT)

    def at(q: dict) -> tuple[float, float]:
        return x0 + (q["V"] + R) / (2 * R) * side, y0 + (1 - (q["A"] + R) / (2 * R)) * side

    for q in sentences:
        x, y = at(q)
        dr.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=FAINT)
    path = [at(q) for q in sentences[: i + 1]]
    if len(path) > 1:
        dr.line(path, fill=MUTED, width=1)
    x, y = path[-1]
    dr.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ACCENT)
    dr.text((x0, y0 + side + 8), f"valence V  (-{R} to {R})", font=fonts["small"], fill=MUTED)
    dr.text((x0, y0 - 20), "arousal A (up)", font=fonts["small"], fill=MUTED)
    # the three parameters
    px0 = 290
    dr.text((px0, y0), "the rule", font=fonts["small"], fill=MUTED)
    rows = [("f  fineness", p["f"], 3.0, 7.0), ("s  depth", p["s"], 0.06, 0.30), ("γ  sharpness", p["gamma"], 0.9, 2.6)]
    for j, (name, v, lo, hi) in enumerate(rows):
        yy = y0 + 30 + 52 * j
        dr.text((px0, yy), f"{name}  {v:.3f}", font=fonts["text"], fill=INK)
        dr.rectangle((px0, yy + 24, px0 + 180, yy + 30), fill=(238, 238, 238))
        dr.rectangle((px0, yy + 24, px0 + 180 * (v - lo) / (hi - lo), yy + 30), fill=ACCENT)
    dr.text((px0, y0 + side + 8), "set by V and A", font=fonts["small"], fill=MUTED)
    # the tile, as height and as lit
    for j, (t, label) in enumerate(zip(tiles, ("height as gray", "lit from the upper left"))):
        tx = 530 + j * (TILE + 30)
        im.paste(t, (tx, y0))
        dr.text((tx, y0 + side + 8), label, font=fonts["small"], fill=MUTED)
    for ax in (254, 492):  # arrows between the panels
        ym = y0 + side / 2
        dr.line((ax, ym, ax + 22, ym), fill=MUTED, width=2)
        dr.polygon([(ax + 28, ym), (ax + 20, ym - 5), (ax + 20, ym + 5)], fill=MUTED)
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    banner(ROOT / "results/work1/common.png", ROOT / "results/work2_sp_color/SPC.png", OUT / "works.png")
    w1c = json.loads((ROOT / "works/conditions/work1.json").read_text())
    w1 = json.loads((ROOT / "results/work1/work1.json").read_text())
    drc = json.loads((ROOT / "works/conditions/work1_drawing.json").read_text())["drawings"]
    d = work1.dimensions(w1c)
    eps = relief.eps_for(d)
    if max(max(abs(p["V"]), abs(p["A"])) for p in w1["sentences"]) > R:
        raise SystemExit("a sentence lies outside the plane drawn")
    sentences = w1["sentences"][:N_SENTENCES]
    for p in sentences:
        if [float(x) for x in params(p["V"], p["A"])] != [p["f"], p["s"], p["gamma"]]:
            raise SystemExit("the parameters differ from the record")
    fonts = {"text": ImageFont.truetype(FONT, 15), "small": ImageFont.truetype(FONT, 13),
             }
    frames = [frame(i, sentences, len(w1["sentences"]), tiles_of(p, d, eps, w1c["relief"]["seed"], drc["elevation_deg"],
                                            drc["azimuth_b_deg"]), fonts) for i, p in enumerate(sentences)]
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(OUT / "process.gif", save_all=True, append_images=pal[1:], duration=1400, loop=0, optimize=True)
    for f in sorted(OUT.iterdir()):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
