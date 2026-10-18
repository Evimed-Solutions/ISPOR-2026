"""Poster figure: the clean baseline and three degraded versions, zoomed where the arms meet.

A redesign of an earlier draft as a compact grid of images only: the row names and labels, and the
column headings, are added on the poster. Rows: baseline, JPEG 95, blur σ 1.5, screenshot.
Columns: (1) pixel change from the clean image at the same DPI, shaded purple over the image so the red and blue
curves stay visible inside it (the baseline, which can't change from itself, is simply its own zoomed pixels); a
callout on JPEG 95 explains its 8×8 blocks; (2) the colour step's arm groups (pixels put in the other arm's group in
amber) with the digitised trace of replicate 1 drawn bold on top, in the colour of the arm the pipeline labelled it
(blue = traced as control, red = traced as treatment; white and dark rings keep it visible on a curve of the same
colour; a line joins the points so flat runs off the curve show) and arrowed callouts naming what the trace did.

Zoom: a square of 0–4 months × survival 0.854–1.02, enlarged by a whole-number factor with no smoothing; the axis
lines at its left and top edges are blanked. Arm groups and the clean
references are computed exactly as in scripts/figure_helpers.py. Pixel change uses the same white-to-purple shade,
laid over the image at 90% on background pixels and 35% on curve pixels. Callout targets are read from the trace (its
survival at a given time, or the top of its largest drop) or, for the JPEG block note, the changed background pixel
nearest a point just above the treatment curve.

The legend and footnote are a separate image, to place on the poster independently of the grid.

Outputs: figures/baseline_comparison.png (the two image columns, 300 dpi), figures/baseline_comparison_legend.png
(270 mm wide, 300 dpi) and figures/baseline_comparison_values.csv
Run: python scripts/figures/baseline_comparison.py   (needs work/pixels from step 3)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from _common import HERE, IMAGES, OUT, RAW, Y_TOP, px_per_month, px_per_survival

import figure_helpers as base

SIDE_MONTHS = 4.0  # square zoom: 0–4 months by the survival span that is as many pixels tall (DPI-independent)
ZOOM = ((0.0, SIDE_MONTHS), (Y_TOP - SIDE_MONTHS * px_per_month(300) / px_per_survival(300), Y_TOP))
base.ZOOM = ZOOM  # full_panel draws its zoom box from this
SPINE_PX = 8  # axis-line band blanked, in image pixels: one JPEG block, which also covers blur and ringing
MM = base.MM  # 300 dpi
WIDTH_MM = 270  # the legend image
TILE_ZOOM = 6  # 300 DPI tiles are enlarged 6× (90 mm); full_baseline.py reads this from the values CSV

ROWS = [  # top to bottom; the row names and labels go on the poster
    "base_clean_300dpi",  # Baseline: 300 DPI PNG
    "jpeg_95",  # JPEG 95: 300 DPI, saved once as JPEG at quality 95
    "blur_1_5",  # Blur σ 1.5: 300 DPI, Gaussian blur
    "workflow_screenshot_slide",  # Screenshot: 120 DPI, blur, JPEG 75
]
# Callouts on the arm-groups tile: text, label box top-left (fractions of the tile), arrow targets.
# A target is (trace label, time in months) for the trace's survival at that time, or (trace label, "drop") for the top of its largest drop.
CALLOUTS = {
    "base_clean_300dpi": [("Control trace", (0.03, 0.83), [("control", 2.2)]),
                          ("Treatment trace", (0.40, 0.04), [("treatment", 3.1)])],
    "blur_1_5": [("Treatment trace\nsits high", (0.55, 0.04), [("treatment", 3.1)])],
    "jpeg_95": [("Traces run flat\nto the legend", (0.50, 0.04), [("control", 3.3), ("treatment", 3.75)])],
    "workflow_screenshot_slide": [("Control trace\nrides the red curve,\nthen drops", (0.19, 0.03),
                                   [("control", 2.0), ("control", "drop")])],
}
BLOCK_NOTE = "JPEG works in 8×8-pixel\nblocks; only blocks that\ntouch a curve change"
BLOCK_ANCHOR = (2.6, 0.935)  # just above the treatment curve, where blocks straddle its edge
INK, MUTED, TEAL, RULE, WHITE = base.INK, base.MUTED, base.TEAL, base.RULE, base.WHITE
FRAME = "#B9B3BE"  # thin neutral tile frame
TRACE_COLOUR = {"control": base.BLUE, "treatment": base.RED}  # by the label the pipeline gave the trace
TRUE_EFFECT = 2.916


def zoom_box(frame: dict) -> tuple[int, int, int, int]:
    """The zoom in image pixels, trimmed to an exact square."""
    x0, y0, x1, _ = base.crops.crop_box(frame, *ZOOM)
    return x0, y0, x1, y0 + (x1 - x0)


def change_over_curves(arr: np.ndarray, clean: np.ndarray, frame: dict) -> np.ndarray:
    """Purple change shading laid over the image, lighter on curve pixels so the curves show inside it.

    The left and top axis lines, and the change around them, are blanked to white (curve pixels are kept)."""
    diff = np.abs(arr.astype(int) - clean.astype(int)).max(axis=2)
    v = np.sqrt(np.clip(diff / base.CHANGE_FULL, 0, 1))
    red, blue = base.arm_colour(arr)
    a = (v * np.where(red | blue, 0.35, 0.9))[..., None]
    out = (arr * (1 - a) + base.hex_rgb(base.PURPLE_DK) * a).round().astype(np.uint8)
    band = SPINE_PX
    rows, cols = np.ogrid[:arr.shape[0], :arr.shape[1]]
    near_axis = (np.abs(cols - frame["left"]) <= band) | (np.abs(rows - frame["top"]) <= band)
    clean_red, clean_blue = base.arm_colour(clean)
    out[near_axis & ~(clean_red | clean_blue)] = 255
    return out


def block_target(arr: np.ndarray, clean: np.ndarray, frame: dict) -> tuple[float, float]:
    """The changed background pixel nearest BLOCK_ANCHOR, in months and survival."""
    x0, y0, x1, y1 = zoom_box(frame)
    diff = np.abs(arr[y0:y1, x0:x1].astype(int) - clean[y0:y1, x0:x1].astype(int)).max(axis=2)
    rows, cols = np.where((diff > 0) & (clean[y0:y1, x0:x1] >= 250).all(axis=2))
    ax, ay = base.crops.to_px(frame, t=BLOCK_ANCHOR[0]) - x0, base.crops.to_px(frame, s=BLOCK_ANCHOR[1]) - y0
    j = np.argmin((cols - ax) ** 2 + (rows - ay) ** 2)
    t = (cols[j] + x0 - frame["left"]) / (frame["right"] - frame["left"]) * base.crops.TAU
    s = (frame["bottom"] - (rows[j] + y0)) / (frame["bottom"] - frame["top"]) * base.crops.Y_TOP
    return float(t), float(s)


def target_point(pts: pd.DataFrame, label: str, when) -> tuple[float, float]:
    if label == "at":  # a fixed point (months, survival)
        return when
    g = pts[pts["raw_curve_label"].str.strip().str.lower() == label]
    g = g[g["time_months"] <= ZOOM[0][1] + 0.5].sort_values(["time_months", "survival"], ascending=[True, False])
    if when == "drop":
        ds = g["survival"].diff()
        j = g.index.get_loc(ds.idxmin())
        return float(g["time_months"].iloc[j]), float(g["survival"].iloc[j - 1])  # top of the drop
    return float(when), float(g[g["time_months"] <= when]["survival"].iloc[-1])


def tile(src: np.ndarray, frame: dict, k: int, pts: pd.DataFrame | None = None, callouts=(), fonts=None):
    x0, y0, x1, y1 = zoom_box(frame)
    size = ((x1 - x0) * k, (y1 - y0) * k)
    out = Image.fromarray(np.ascontiguousarray(src[y0:y1, x0:x1])).resize(size, Image.NEAREST).convert("RGBA")
    if pts is None:
        return out.convert("RGB")
    ss = 3
    to = lambda t, s: ((base.crops.to_px(frame, t=t) - x0 + 0.5) * k * ss, (base.crops.to_px(frame, s=s) - y0 + 0.5) * k * ss)
    over = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    r = 1.5 * MM * ss  # 3 mm dots
    for arm, g in base.traces(pts):  # arm-coloured line in white and dark casings, through every point
        xy = [to(t, s) for t, s in zip(g["time_months"], g["survival"])]
        for colour, wid in ((INK, 2.0), (WHITE, 1.5), (TRACE_COLOUR.get(arm, INK), 0.85)):
            d.line(xy, fill=colour, width=round(wid * MM * ss), joint="curve")
    for arm, g in base.traces(pts):
        for t, s in zip(g["time_months"], g["survival"]):
            px, py = to(t, s)
            if -r <= px <= size[0] * ss + r and -r <= py <= size[1] * ss + r:
                for colour, rr in ((INK, r), (WHITE, r - 0.25 * MM * ss), (TRACE_COLOUR.get(arm, INK), r - 0.6 * MM * ss)):
                    d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=colour)
    # callouts: white label box, cased arrow to each target
    f = fonts["callout"]
    fs = ImageFont.truetype(f.path, f.size * ss)
    head = 2.6 * MM * ss
    for text, (fx, fy), targets in callouts:
        lines = text.split("\n")
        tw = max(d.textlength(line, font=fs) for line in lines)
        padx, pady, margin = 1.2 * MM * ss, 0.7 * MM * ss, 1.5 * MM * ss
        bw, bh = tw + 2 * padx, len(lines) * fs.size * 1.2 + 2 * pady
        bx0 = min(fx * size[0] * ss, size[0] * ss - bw - margin)  # keep the box inside the tile
        by0 = min(fy * size[1] * ss, size[1] * ss - bh - margin)
        box = (bx0, by0, bx0 + bw, by0 + bh)
        for who, when in targets:
            tx, ty = to(*target_point(pts, who, when))
            sx = min(max(tx, box[0]), box[2])  # leave the box from the side nearest the target
            sy = box[3] if ty > box[3] else box[1] if ty < box[1] else (box[1] + box[3]) / 2
            ang = math.atan2(ty - sy, tx - sx)
            ex, ey = tx - math.cos(ang) * 1.6 * r, ty - math.sin(ang) * 1.6 * r  # stop short of the dot
            hx, hy = ex - math.cos(ang) * head, ey - math.sin(ang) * head
            wing = [(hx - math.sin(ang) * head * 0.5, hy + math.cos(ang) * head * 0.5),
                    (hx + math.sin(ang) * head * 0.5, hy - math.cos(ang) * head * 0.5)]
            for colour, wid, grow in ((WHITE, 1.4, 0.5 * MM * ss), (INK, 0.55, 0)):
                d.line([(sx, sy), (hx, hy)], fill=colour, width=round(wid * MM * ss))
                tip = (ex + math.cos(ang) * grow, ey + math.sin(ang) * grow)
                d.polygon([tip, *[(wx - math.cos(ang) * grow, wy - math.sin(ang) * grow) for wx, wy in wing]],
                          fill=colour)
        d.rounded_rectangle(box, radius=0.8 * MM * ss, fill=(255, 255, 255, 235), outline=INK,
                            width=round(0.25 * MM * ss))
        for n, line in enumerate(lines):
            d.text((box[0] + padx, box[1] + pady + n * fs.size * 1.2), line, fill=INK, font=fs)
    out.alpha_composite(over.resize(size, Image.LANCZOS))
    return out.convert("RGB")


def main() -> None:
    errors = pd.read_csv(OUT / "results_by_error_type.csv").set_index("image_id")
    quality = pd.read_csv(OUT / "image_quality_metrics.csv").set_index("image_id")
    points = pd.read_csv(RAW / "kmgpt_curve_points.csv.gz", usecols=lambda c: c != "notes")

    pad, tile_gap, row_gap = (round(v * MM) for v in (5, 4, 6))
    fonts = {"callout": base.font(16, True)}

    rows, records = [], []
    ref_w = None
    for image_id in ROWS:
        arr = np.asarray(Image.open(IMAGES / f"{image_id}.png").convert("RGB"))
        ref = quality.loc[image_id, "compared_with"]
        clean = arr if pd.isna(ref) else np.asarray(Image.open(IMAGES / f"{ref}.png").convert("RGB"))
        frame = base.crops.find_frame(arr)
        x0, _, x1, _ = zoom_box(frame)
        if ref_w is None:  # the 300 DPI tiles set the size; lower-DPI tiles get the nearest whole factor to match
            ref_w = TILE_ZOOM * (x1 - x0)
        k = max(1, round(ref_w / (x1 - x0)))
        pts = points[points["run_id"] == f"{image_id}_rep01"]
        groups, gstats = base.groups_rgb(base.arm_groups(image_id, arr.shape[:2]), clean)
        if pd.isna(ref):  # the reference can't change from itself, so this is just its zoomed pixels
            notes = [("Clean image, zoomed\n(no pixel change)", (0.50, 0.04), [])]
        elif image_id == "jpeg_95":
            notes = [(BLOCK_NOTE, (0.42, 0.03), [("at", block_target(arr, clean, frame))])]
        else:
            notes = []
        first = tile(change_over_curves(arr, clean, frame), frame, k, pts.iloc[0:0] if notes else None, notes, fonts)
        tiles = [first, tile(groups, frame, k, pts, CALLOUTS.get(image_id, ()), fonts)]
        e = errors.loc[image_id]
        rows.append(tiles)
        records.append({"image_id": image_id, "clean_reference": image_id if pd.isna(ref) else ref, "zoom_scale": k,
                        "scored_runs": e["scored_runs"], "te_error": e["te_error"],
                        "te_error_pct": e["te_error_pct_of_true_effect"],
                        **{key: round(v, 2) for key, v in gstats.items()},
                        "callouts": " | ".join(c[0].replace("\n", " ") for c in CALLOUTS.get(image_id, ()))})

    f_leg, f_foot = base.font(18), base.font(15)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    tl = lambda text, fnt: round(probe.textlength(text, font=fnt))
    lh = lambda fnt: round(1.25 * fnt.size)

    def wrap(text, fnt, max_w):
        lines, line = [], ""
        for word in text.split(" "):
            trial = (line + " " + word).strip()
            if tl(trial, fnt) <= max_w or not line:
                line = trial
            else:
                lines.append(line); line = word
        return lines + [line]

    def block(d, x, y, lines, fnt, fill):
        for line in lines:
            d.text((x, y), line, fill=fill, font=fnt); y += lh(fnt)
        return y

    slot = max(t.width for tiles in rows for t in tiles)  # the two image columns, sized to the widest tile
    width = 2 * pad + 2 * slot + tile_gap
    xs = [pad + j * (slot + tile_gap) for j in range(2)]
    row_h = [max(t.height for t in tiles) for tiles in rows]
    height = pad + sum(row_h) + row_gap * (len(rows) - 1) + pad
    fig = Image.new("RGB", (width, height), WHITE)
    d = ImageDraw.Draw(fig)

    y = pad
    for i, (tiles, rh) in enumerate(zip(rows, row_h)):
        records[i].update(row_top_mm=round(y / MM, 2), row_height_mm=round(rh / MM, 2))  # places the poster labels
        for j, (x, t) in enumerate(zip(xs, tiles)):
            tx = x + (slot - t.width) // 2
            fig.paste(t, (tx, y))
            d.rectangle([tx - 3, y - 3, tx + t.width + 2, y + t.height + 2], outline=FRAME, width=3)
        y += rh
        if i < len(rows) - 1:
            d.line([(pad, y + row_gap // 2), (width - pad, y + row_gap // 2)], fill=RULE, width=3)
            y += row_gap

    # legend and footnote: a separate image, WIDTH_MM wide, placed on the poster independently
    width = round(WIDTH_MM * MM)
    foot = ("Zoom: 0–4 months, survival 0.85–1.02, enlarged by a whole-number factor with no smoothing; axis lines removed. "
            "Pixel change = "
            "largest R, G or B difference from a clean image at the same DPI, shaded purple over the image. JPEG "
            "compresses each 8×8-pixel block on its own and rounds away fine detail, so its changes come in blocks: 99% "
            "of changed blocks touch a curve, and two-thirds of changed pixels moved by only 1–3 of 255 levels. That is "
            "invisible, but enough to tip red's hue past 0°/360°. Arm groups: each pixel in the colour of the group the "
            "algorithm put it in. Trace: replicate 1, coloured by the arm the pipeline labelled it. TE = digitised minus "
            f"true 36-month RMST difference (true {TRUE_EFFECT:.2f} months).")
    foot_l = wrap(foot, f_foot, width - 2 * pad)
    key = Image.new("RGB", (width, pad + 2 * lh(f_leg) + round(4 * MM) + len(foot_l) * lh(f_foot) + pad), WHITE)
    d = ImageDraw.Draw(key)
    y = pad
    lx, cy = pad, y + lh(f_leg) // 2
    space = round(6 * MM)
    r_ = round(1.5 * MM)
    for arm in ("control", "treatment"):  # the same ringed, arm-coloured dot as in the tiles
        for colour, rr in ((INK, r_), (WHITE, r_ - round(0.25 * MM)), (TRACE_COLOUR[arm], r_ - round(0.6 * MM))):
            d.ellipse([lx + r_ - rr, cy - rr, lx + r_ + rr, cy + rr], fill=colour)
        d.text((lx + 2 * r_ + 14, y), f"Traced as {arm}", fill=INK, font=f_leg)
        lx += 2 * r_ + 14 + tl(f"Traced as {arm}", f_leg) + space
    sw = round(0.8 * f_leg.size)
    d.rectangle([lx, cy - sw // 2, lx + sw, cy + sw // 2], fill=base.AMBER, outline=MUTED, width=2)
    d.text((lx + sw + 14, y), "Put in the other arm's group", fill=INK, font=f_leg)
    right1 = lx + sw + 14 + tl("Put in the other arm's group", f_leg)
    y2 = y + lh(f_leg)
    cy2 = y2 + lh(f_leg) // 2
    d.text((pad, y2), "Pixel change:", fill=INK, font=f_leg)
    bx, bw = pad + tl("Pixel change:", f_leg) + 20, round(40 * MM)
    for x in range(bw):
        v = np.sqrt(x / (bw - 1))
        d.line([(bx + x, cy2 - sw // 2), (bx + x, cy2 + sw // 2)],
               fill=tuple(int(round(255 * (1 - v) + c * v)) for c in base.hex_rgb(base.PURPLE_DK)))
    d.rectangle([bx, cy2 - sw // 2, bx + bw, cy2 + sw // 2], outline=MUTED, width=2)
    d.text((bx + bw + 16, y2), "0 → 8+ of 255 levels", fill=INK, font=f_leg)
    if max(right1, bx + bw + 16 + tl("0 → 8+ of 255 levels", f_leg)) > width - pad:
        raise SystemExit("legend overflows")
    y = y2 + lh(f_leg) + round(4 * MM)
    block(d, pad, y, foot_l, f_foot, MUTED)

    for name, img in (("baseline_comparison.png", fig), ("baseline_comparison_legend.png", key)):
        out = base.FIG_DIR / name
        img.save(out, dpi=(base.DPI_OUT, base.DPI_OUT), optimize=True)
        print(f"wrote {out} ({img.width} × {img.height} px = {img.width / MM:.0f} × {img.height / MM:.0f} mm at "
              f"{base.DPI_OUT} dpi, {out.stat().st_size / 1e6:.1f} MB)")
    pd.DataFrame(records).to_csv(base.FIG_DIR / "baseline_comparison_values.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
