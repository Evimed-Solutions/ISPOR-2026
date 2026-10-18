"""Poster figure: the full baseline KM plot redrawn at poster size, with its censor marks and the digitised points.

The plot is the study's own figure (render_pdf in scripts/core/generate_images.py, image base_clean_300dpi) drawn
again by the same matplotlib code from the same data (truth_km_curve.csv, base_trial.csv), so its geometry, fonts,
colours and legend are the test image's. White margins are trimmed and the PNG is saved at the scale that makes it
WIDTH_MM wide at 300 dpi; the SVG is the same figure as vector paths, for enlarging without loss.

On top, as in baseline_comparison.py: the digitised trace of replicate 1 (kmgpt_curve_points.csv.gz) as ringed
dots in the colour of the arm the pipeline labelled them, joined by a cased line, so flat runs show. Their sizes are
that figure's divided by its zoom factor, so a region of this plot enlarged to the comparison tiles' scale shows the
same dots at the same size. Censor marks keep the study's positions (every censored patient at the KM survival; all
fall after 30 months) and its "+" marker, drawn CENSOR_SIZE times taller and CENSOR_EDGE times thicker so they read
on the poster. A key under the axes names both.

The zoom-box version adds a teal outline of the comparison figure's zoom (0–4 months × survival 0.854–1.02).

Check: the plot alone at the study's own size, rasterised with pdftoppm at 300 dpi as the study did, is compared
pixel by pixel with data/images/base_clean_300dpi.png.

Outputs: figures/full_plot_baseline.png and full_plot_baseline_zoombox.png (WIDTH_MM wide, 300 dpi), the same two as
.svg, and figures/full_plot_baseline_validation.json
Run: python full_baseline.py   (needs pdftoppm on PATH, as generate_images.py does)
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from PIL import Image

from _common import HERE, IMAGES, RAW, ROOT, TRUTH  # _common also puts ROOT/scripts on the import path
from km_synthetic_utils import ARM_CONTROL, ARMS, step_survival_at

import baseline_comparison as comparison
base = comparison.base

IMAGE_ID = "base_clean_300dpi"
RUN_ID = f"{IMAGE_ID}_rep01"
WIDTH_MM = 270  # the comparison figure's column
DPI_OUT = 300
PAD_MM = 3  # white margin kept round the trimmed figure, at poster size
CENSOR_SIZE, CENSOR_EDGE = 1.8, 1.3  # censor marks relative to the study's (5 pt "+", 0.9 pt stroke)
BOX_MM = 0.8  # zoom-box outline, at poster size
MAX_CENSOR_TICKS = 45  # the study thins censor marks to this many per arm (both arms have fewer)
INK, WHITE, TEAL = base.INK, base.WHITE, base.TEAL
TRACE_COLOUR = comparison.TRACE_COLOUR
PT_PER_MM = 72 / 25.4


def draw_plot(cfg: dict, trial: pd.DataFrame, km: pd.DataFrame, censor_size: float = 1.0, censor_edge: float = 1.0):
    """The study's figure for base_clean_300dpi (render_pdf, no risk table, default palette), censor marks scaled."""
    r = cfg["render"]
    palette = r["colour_palettes"]["default"]
    horizon = float(cfg["trial"]["admin_followup_months"])
    fig = plt.figure(figsize=(float(r["figure_width_inches"]), float(r["figure_height_inches"])))
    ax = fig.add_axes(r["axis_rect"]["absent"])
    for arm in ARMS:
        arm_df = km[km["arm"] == arm].sort_values("time_months")
        ax.step(arm_df["time_months"], arm_df["survival"], where="post", lw=float(r["line_width_points"]),
                color=palette[arm], label="Control" if arm == ARM_CONTROL else "Treatment")
        cens = trial[(trial["arm"] == arm) & (trial["event"] == 0)]["observed_time_months"].to_numpy(float)
        cens = np.sort(cens[(cens >= 0) & (cens <= horizon)])
        if len(cens) > MAX_CENSOR_TICKS:
            cens = cens[np.linspace(0, len(cens) - 1, MAX_CENSOR_TICKS).astype(int)]
        surv = step_survival_at(arm_df, cens)
        ax.plot(cens, surv, linestyle="none", marker="+", markersize=float(r["censor_tick_size_points"]),
                markeredgewidth=0.9, color=palette[arm])
        if censor_size != 1 or censor_edge != 1:  # a taller vertical tick over the "+"; its horizontal arm stays as tested
            ax.plot(cens, surv, linestyle="none", marker="|", markersize=float(r["censor_tick_size_points"]) * censor_size,
                    markeredgewidth=0.9 * censor_edge, color=palette[arm])
    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(r["x_ticks"])
    ax.set_yticks(r["y_ticks"])
    ax.set_xlabel("Months")
    ax.set_ylabel("Overall survival")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False)
    return fig, ax


def ringed_dot(arm: str, scale: float, size: dict) -> tuple:
    """A legend handle drawn like the trace dots: dark ring, white ring, arm-coloured centre."""
    return tuple(Line2D([], [], linestyle="none", marker="o", markersize=d * scale, markeredgewidth=0, markerfacecolor=c)
                 for c, d in ((INK, size["dot"]), (WHITE, size["dot_white"]), (TRACE_COLOUR[arm], size["dot_fill"])))


def add_trace(ax, pts: pd.DataFrame, size: dict) -> None:
    """Draw the trace as in baseline_comparison.tile: cased lines for both arms, then ringed dots."""
    traces = list(base.traces(pts))
    for arm, g in traces:
        for colour, wid in ((INK, size["line"]), (WHITE, size["line_white"]), (TRACE_COLOUR.get(arm, INK), size["line_fill"])):
            ax.plot(g["time_months"], g["survival"], color=colour, lw=wid, solid_joinstyle="round",
                    solid_capstyle="round", zorder=4)
    for arm, g in traces:  # dark, white and arm-coloured circle per dot, dot by dot, so each dot keeps its rings
        n = len(g)
        ax.scatter(np.repeat(g["time_months"].to_numpy(), 3), np.repeat(g["survival"].to_numpy(), 3),
                   s=np.tile([size["dot"] ** 2, size["dot_white"] ** 2, size["dot_fill"] ** 2], n),
                   c=[INK, WHITE, TRACE_COLOUR.get(arm, INK)] * n, linewidths=0, zorder=5)


def add_key(fig, cfg: dict, size: dict) -> None:
    """Key under the axes: each arm's censor mark and digitised dot, drawn as in the plot (dots enlarged to read)."""
    palette = cfg["render"]["colour_palettes"]["default"]
    ms = float(cfg["render"]["censor_tick_size_points"])
    censors = tuple(tuple(Line2D([], [], linestyle="none", marker=m, markersize=s, markeredgewidth=w, color=palette[a])
                          for m, s, w in (("+", ms, 0.9), ("|", ms * CENSOR_SIZE, 0.9 * CENSOR_EDGE)))
                    for a in ARMS)
    dots = tuple(ringed_dot(a, 4, size) for a in ARMS)
    overlay = HandlerTuple(ndivide=1)  # the parts of one mark, drawn on top of each other
    fig.legend([censors, dots], ["Censored", "Digitised points"],  # replicate 1; all three baseline runs match
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6), **{h: overlay for h in censors + dots}},
               loc="upper center", bbox_to_anchor=(0.54, 0.06), ncol=2, frameon=False, columnspacing=2.5,
               handlelength=2.6)


def check_geometry(cfg: dict, trial: pd.DataFrame, km: pd.DataFrame) -> dict:
    """Redraw the plot at the study's size, rasterise it as the study did and compare it with the study image."""
    fig, _ = draw_plot(cfg, trial, km)
    with tempfile.TemporaryDirectory(prefix="full_plot_check_") as tmp:
        pdf = Path(tmp) / "check.pdf"
        fig.savefig(pdf, metadata={"Creator": "kmgpt_synthetic_image_quality_v3"})
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "300", str(pdf), str(Path(tmp) / "check")], check=True)
        mine = np.asarray(Image.open(Path(tmp) / "check.png").convert("RGB"))
    plt.close(fig)
    study = np.asarray(Image.open(IMAGES / f"{IMAGE_ID}.png").convert("RGB"))
    if mine.shape != study.shape:
        return {"same_size_as_study_image": False, "redrawn_px": list(mine.shape[:2]), "study_px": list(study.shape[:2])}
    diff = np.abs(mine.astype(int) - study.astype(int)).max(axis=2)
    return {"same_size_as_study_image": True, "pixels_differing": int((diff > 0).sum()),
            "max_rgb_difference": int(diff.max())}


def scale_svg(path: Path, scale: float) -> None:
    """Set the SVG's width and height to poster size; the viewBox, and so the drawing, is unchanged."""
    text = path.read_text(encoding="utf-8")
    head = re.search(r"<svg[^>]*>", text).group(0)
    new = re.sub(r'(width|height)="([\d.]+)pt"', lambda m: f'{m.group(1)}="{float(m.group(2)) * scale:.2f}pt"', head)
    path.write_text(text.replace(head, new, 1), encoding="utf-8")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config/study.yaml").read_text(encoding="utf-8"))
    trial = pd.read_csv(TRUTH / "base_trial.csv")
    km = pd.read_csv(TRUTH / "truth_km_curve.csv")
    points = pd.read_csv(RAW / "kmgpt_curve_points.csv.gz", usecols=lambda c: c != "notes")
    pts = points[points["run_id"] == RUN_ID]
    zoom_k = int(pd.read_csv(base.FIG_DIR / "baseline_comparison_values.csv")
                 .set_index("image_id").loc[IMAGE_ID, "zoom_scale"])

    check = check_geometry(cfg, trial, km)
    print("geometry check against the study image:", check)

    # Trace sizes: the comparison tile's (mm at its zoom k) divided by k gives points in this figure, which is the
    # study's own figure size; the poster scale below enlarges everything together.
    mm = lambda v: v / zoom_k * PT_PER_MM
    size = {"dot": mm(3.0), "dot_white": mm(2.5), "dot_fill": mm(1.8),
            "line": mm(2.0), "line_white": mm(1.5), "line_fill": mm(0.85)}

    fig, ax = draw_plot(cfg, trial, km, CENSOR_SIZE, CENSOR_EDGE)
    add_trace(ax, pts, size)
    add_key(fig, cfg, size)
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    scale = (WIDTH_MM - 2 * PAD_MM) / 25.4 / tight.width  # the margin is PAD_MM at poster size
    pad = PAD_MM / 25.4 / scale
    bbox = Bbox.from_extents(tight.x0 - pad, tight.y0 - pad, tight.x1 + pad, tight.y1 + pad)

    out = {}
    for variant in ("", "_zoombox"):
        if variant:
            (t0, t1), (s0, s1) = comparison.ZOOM
            ax.add_patch(Rectangle((t0, s0), t1 - t0, s1 - s0, fill=False, edgecolor=TEAL,
                                   lw=BOX_MM / scale * PT_PER_MM, joinstyle="miter", clip_on=False, zorder=6))
        stem = base.FIG_DIR / f"full_plot_baseline{variant}"
        fig.savefig(stem.with_suffix(".png"), dpi=DPI_OUT * scale, bbox_inches=bbox, pil_kwargs={"dpi": (DPI_OUT, DPI_OUT)})
        fig.savefig(stem.with_suffix(".svg"), bbox_inches=bbox)
        scale_svg(stem.with_suffix(".svg"), scale)
        with Image.open(stem.with_suffix(".png")) as im:
            out[stem.name] = {"px": list(im.size), "mm": [round(v / DPI_OUT * 25.4, 1) for v in im.size],
                              "dpi": [round(v) for v in im.info["dpi"]]}
        print(f"wrote {stem.name}.png {out[stem.name]} and .svg")
    plt.close(fig)

    censored = trial[trial["event"] == 0].groupby("arm")["observed_time_months"].agg(["size", "min"])
    report = {"image_id": IMAGE_ID, "run_id": RUN_ID, "geometry_check": check, "outputs": out,
              "poster_scale_vs_study_figure": round(scale, 4), "comparison_zoom_factor": zoom_k,
              "digitised_points": {k: int(v) for k, v in pts.groupby("raw_curve_label").size().items()},
              "censor_marks": {a: {"count": int(r["size"]), "first_month": round(float(r["min"]), 2)}
                               for a, r in censored.iterrows()},
              "censor_mark_scale": {"height": CENSOR_SIZE, "stroke": CENSOR_EDGE},
              "zoom_box": {"months": list(comparison.ZOOM[0]), "survival": [round(v, 4) for v in comparison.ZOOM[1]]}}
    path = base.FIG_DIR / "full_plot_baseline_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
