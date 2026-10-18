"""Shared colour grouping, trace styling, and font helpers for the retained figures."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import ImageFont
import image_geometry as crops
from _common import FOLDER, WORK

FIG_DIR = FOLDER / "figures"
DPI_OUT = 300
MM = DPI_OUT / 25.4
ZOOM = ((0.0, 4.0), (0.854, 1.02))
CHANGE_FULL = 8
INK, PURPLE_DK, MUTED, TEAL = "#1E012D", "#450369", "#545050", "#11997C"
RULE, WHITE, BLUE, RED, AMBER = "#D9D4DC", "#FFFFFF", "#1f77b4", "#b91c1c", "#E69F00"
FONT_DIR = str(Path(os.environ.get("ISPOR_FONT_DIR", str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")))) + "/"

def font(pt: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_DIR + ("arialbd.ttf" if bold else "arial.ttf"), round(pt / 72 * DPI_OUT))


def hex_rgb(c: str) -> np.ndarray:
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def arm_colour(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Red and blue curve pixels by HSL hue (any lightness short of white), as in a clean image."""
    rgb = arr.astype(float) / 255
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    delta, light = mx - mn, (mx + mn) / 2
    sat = np.where(delta > 0, delta / np.where(1 - np.abs(2 * light - 1) == 0, 1, 1 - np.abs(2 * light - 1)), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.where(delta == 0, 1, delta)
    hue = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60
    curve = (delta > 0) & (sat >= 0.15) & (light < 0.98)
    return curve & ((hue <= 20) | (hue >= 340)), curve & (hue >= 180) & (hue < 250)


def arm_groups(image_id: str, shape: tuple[int, int]) -> np.ndarray:
    """0 = not clustered, 1 = blue group, 2 = red group (the group whose median hue is nearest 205° is blue)."""
    px = pd.read_csv(WORK / "pixels" / f"{image_id}_pixels.csv")
    blue_group = px.groupby("group")["h"].median().sub(205).abs().idxmin()
    out = np.zeros(shape, dtype=np.uint8)
    rows, cols = shape[0] - px["arr_row"].to_numpy(), px["arr_col"].to_numpy() - 1  # rows are counted from the bottom
    out[rows, cols] = np.where(px["group"].to_numpy() == blue_group, 1, 2)
    return out


def groups_rgb(groups: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, dict]:
    red, blue = arm_colour(clean)
    moved = (red & (groups == 1)) | (blue & (groups == 2))
    out = np.full((*groups.shape, 3), 255, dtype=np.uint8)
    out[groups == 1], out[groups == 2], out[moved] = hex_rgb(BLUE), hex_rgb(RED), hex_rgb(AMBER)
    stats = {"red_pixels_in_blue_group_pct": 100 * (red & (groups == 1)).sum() / max((red & (groups > 0)).sum(), 1),
             "blue_pixels_in_red_group_pct": 100 * (blue & (groups == 2)).sum() / max((blue & (groups > 0)).sum(), 1)}
    return out, stats


def traces(pts: pd.DataFrame):
    for label, g in pts.groupby("raw_curve_label"):
        yield label.strip().lower(), g.sort_values(["time_months", "survival"], ascending=[True, False])
