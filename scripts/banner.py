"""Draw the README banner and the icon in the house style (dark, contour rings).

The same style as the other projects of the author: a dark slate background, a logo of
concentric contours from the cold blue outside to the hot red core, and the name in bold
white.  Here the contours are the isotherms of the silo in section: taller than wide,
cold at the insulation, hot at the core.

    python scripts/banner.py      # writes assets/banner.png and assets/icon.png
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (15, 27, 36)
RINGS = ((23, 100, 230), (22, 150, 235), (245, 120, 30), (220, 30, 30))
TEXT = (255, 255, 255)
SCALE = 4                                       # drawn large, then filtered down
ROOT = Path(__file__).resolve().parents[1]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("Artifakt Element Black.ttf", "segoeuib.ttf", "arialbd.ttf"):
        for folder in (Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/"
                       "Windows/Fonts"):
            path = folder / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _logo(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float) -> None:
    """The silo in section: four rounded isotherms, cold outside, hot at the core.

    ``radius`` is the half height of the outer contour; the contours are taller than
    wide, like the vessel, and each one is a rounded rectangle whose corners grow
    rounder towards the core, as the isotherms of a bed heated from inside do.
    """
    stroke = radius * 0.11
    inset = 2.05 * stroke                      # the same gap between every two contours
    for index, colour in enumerate(RINGS):
        half_h = radius - index * inset
        half_w = 0.86 * radius - index * inset
        corner = half_w * min(0.55 + 0.15 * index, 1.0)
        draw.rounded_rectangle((cx - half_w, cy - half_h, cx + half_w, cy + half_h),
                               radius=corner, outline=colour, width=int(stroke))


def banner(path: Path, width: int = 1280, height: int = 320,
           title: str = "Thermal Battery Simulator") -> None:
    w, h = width * SCALE, height * SCALE
    image = Image.new("RGB", (w, h), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _logo(draw, 0.155 * w, 0.5 * h, 0.40 * h)
    font = _font(int(0.30 * h))
    box = draw.textbbox((0, 0), title, font=font)
    x = 0.30 * w
    available = 0.96 * w - x
    if box[2] - box[0] > available:
        font = _font(int(0.30 * h * available / (box[2] - box[0])))
        box = draw.textbbox((0, 0), title, font=font)
    y = 0.5 * h - 0.5 * (box[1] + box[3])
    draw.text((x, y), title, font=font, fill=TEXT)
    image.resize((width, height), Image.LANCZOS).save(path)


def icon(path: Path, size: int = 1024) -> None:
    s = size * SCALE
    image = Image.new("RGB", (s, s), BACKGROUND)
    _logo(ImageDraw.Draw(image), 0.5 * s, 0.5 * s, 0.4 * s)
    image.resize((size, size), Image.LANCZOS).save(path)


if __name__ == "__main__":
    (ROOT / "assets").mkdir(exist_ok=True)
    banner(ROOT / "assets" / "banner.png")
    icon(ROOT / "assets" / "icon.png")
    print("wrote assets/banner.png and assets/icon.png")
