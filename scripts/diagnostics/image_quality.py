"""How each factor changes the pixels, measured against a clean render at the same DPI.

Outputs (results/diagnostics):
  image_quality_metrics.csv  per image: size, distinct colours, pixels changed, mean change, MSE, PSNR, SSIM
  dpi_geometry.csv           per DPI: pixel size of lines, ticks, months and survival steps
  blur_kernel.csv            per blur level: physical size, reach, how much colour survives at line centres
  jpeg_quantisation.csv      per JPEG quality: Pillow's actual quantisation steps
  recompression_steps.csv    change between successive recompression levels
  palette_colours.csv        per palette: HSL (as SurvdigitizeR computes it), luma, CIE76 colour differences

Requires scikit-image (MSE, PSNR, SSIM). SSIM uses Wang et al. (2004) settings:
Gaussian window sigma 1.5, population covariance, averaged over the RGB channels.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from _common import (TRUTH, CENSOR_STROKE_PT, CLEAN_REFERENCE, FACTOR_ORDER, IMAGES, LINE_WIDTH_PT, RESULTS,
                     TAU, manifest, px_per_month, px_per_survival, write)


def load(image_id: str) -> np.ndarray:
    return np.asarray(Image.open(IMAGES / f"{image_id}.png").convert("RGB"))


def distinct_colours(arr: np.ndarray) -> int:
    return len(Image.fromarray(arr).getcolors(2 ** 24))


def compare(img: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = np.abs(img.astype(int) - ref.astype(int))
    return {
        "pixels_changed_pct": 100 * float((diff.max(axis=2) > 0).mean()),
        "mean_change": float(diff.mean()),
        "mse": float(mean_squared_error(ref, img)),
        "psnr_db": float(peak_signal_noise_ratio(ref, img, data_range=255)),
        "ssim": float(structural_similarity(ref, img, channel_axis=-1, data_range=255,
                                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
    }


def image_quality_metrics() -> pd.DataFrame:
    meta = manifest()
    outcome = pd.read_csv(RESULTS / "kmgpt_rmst_error_by_condition.csv").set_index("image_id")
    rows = []
    for image_id in FACTOR_ORDER:
        dpi = int(meta.loc[image_id, "dpi"])
        img = load(image_id)
        reference = CLEAN_REFERENCE[dpi]
        row = {"image_id": image_id, "factor": meta.loc[image_id, "factor"], "dpi": dpi,
               "width_px": img.shape[1], "height_px": img.shape[0], "distinct_colours": distinct_colours(img)}
        # Pixel comparison is meaningful only for degradations applied after rendering:
        # each DPI image is its own clean render, and palettes differ by design.
        comparable = image_id != reference and meta.loc[image_id, "factor"] not in ("colour_separability",)
        row["compared_with"] = reference if comparable else ""
        if comparable:
            row.update(compare(img, load(reference)))
        o = outcome.loc[image_id]
        row["scored_runs"] = f"{int(o['scored_runs'])}/{int(o['total_runs'])}"
        row["mean_abs_arm_error"] = o["mean_abs_arm_rmst_error_months_mean"] if o["scored_runs"] else np.nan
        row["te_error"] = o["rmst_difference_error_months_mean"] if o["scored_runs"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def dpi_geometry() -> pd.DataFrame:
    rows = []
    for dpi in (300, 200, 150, 120, 90, 75):
        height, width = load(CLEAN_REFERENCE[dpi]).shape[:2]  # actual raster size (pdftoppm rounds up)
        rows.append({
            "dpi": dpi, "width_px": width, "height_px": height,
            "curve_line_px": LINE_WIDTH_PT / 72 * dpi, "censor_tick_stroke_px": CENSOR_STROKE_PT / 72 * dpi,
            "px_per_month": px_per_month(dpi), "px_per_0_01_survival": px_per_survival(dpi) * 0.01,
            "rmst_months_per_px_vertical_offset": TAU / px_per_survival(dpi),
        })
    return pd.DataFrame(rows)


def blur_kernel() -> pd.DataFrame:
    line_px = LINE_WIDTH_PT / 72 * 300
    tick_px = CENSOR_STROKE_PT / 72 * 300
    rows = []
    for sigma in (0.3, 0.7, 1.1, 1.5):
        rows.append({
            "sigma_px": sigma, "physical_sigma_mm": sigma / 300 * 25.4, "reach_3sigma_px": 3 * sigma,
            # Centre of a line of width w convolved with a Gaussian keeps erf(w / (2*sqrt(2)*sigma))
            "centre_kept_curve_pct": 100 * math.erf(line_px / (2 * math.sqrt(2) * sigma)),
            "centre_kept_tick_pct": 100 * math.erf(tick_px / (2 * math.sqrt(2) * sigma)),
        })
    return pd.DataFrame(rows)


def jpeg_quantisation() -> pd.DataFrame:
    probe = Image.new("RGB", (16, 16), "white")
    rows = []
    for quality in (95, 85, 75, 65, 55):
        buffer = io.BytesIO()
        probe.save(buffer, format="JPEG", quality=quality, subsampling=0)  # same call as the generator
        buffer.seek(0)
        tables = Image.open(buffer).quantization
        luma, chroma = list(tables[0]), list(tables[1])
        rows.append({"quality": quality, "scale_pct": 5000 / quality if quality < 50 else 200 - 2 * quality,
                     "luma_dc_step": luma[0], "luma_max_step": max(luma),
                     "chroma_dc_step": chroma[0], "chroma_max_step": max(chroma)})
    return pd.DataFrame(rows)


def recompression_steps() -> pd.DataFrame:
    levels = {k: load(f"recompression_{k}").astype(int) for k in (1, 3, 5)}
    return pd.DataFrame([
        {"from": "recompression_1", "to": "recompression_3", "mean_change": float(np.abs(levels[1] - levels[3]).mean())},
        {"from": "recompression_3", "to": "recompression_5", "mean_change": float(np.abs(levels[3] - levels[5]).mean())},
        {"from": "jpeg_55", "to": "recompression_1", "mean_change": float(np.abs(load("jpeg_55").astype(int) - levels[1]).mean())},
    ])


def hsl_like_survdigitizer(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Hue in degrees (rounded, linear), saturation and lightness 0-1, as SurvdigitizeR::img_read."""
    r, g, b = (v / 255 for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    l = (mx + mn) / 2
    if delta == 0:
        return 0.0, 0.0, l
    if mx == r:
        h = ((g - b) / delta) % 6
    elif mx == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4
    s = delta / (1 - abs(2 * l - 1))
    return float(round(h * 60)), s, l


def palette_colours() -> pd.DataFrame:
    palettes = {
        "default": ("#1f77b4", "#b91c1c"), "easy": ("#1f77b4", "#2ca02c"),
        "hard": ("#fef002", "#fdf577"), "greyscale (default converted)": ("#646464", "#4b4b4b"),
    }
    to_rgb = lambda hx: tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))
    lab = lambda hx: rgb2lab(np.array([[to_rgb(hx)]], dtype=float) / 255)[0, 0]
    white = lab("#ffffff")
    rows = []
    for name, (control, treatment) in palettes.items():
        row = {"palette": name, "control": control, "treatment": treatment}
        for arm, hx in (("control", control), ("treatment", treatment)):
            h, s, l = hsl_like_survdigitizer(to_rgb(hx))
            r, g, b = to_rgb(hx)
            row.update({f"{arm}_hue": h, f"{arm}_saturation": round(s, 3), f"{arm}_lightness": round(l, 3),
                        f"{arm}_luma": round(0.299 * r + 0.587 * g + 0.114 * b),
                        f"{arm}_delta_e_vs_white": float(np.linalg.norm(lab(hx) - white))})
        row["delta_e_between_arms"] = float(np.linalg.norm(lab(control) - lab(treatment)))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    write(image_quality_metrics(), "image_quality_metrics.csv")
    write(dpi_geometry(), "dpi_geometry.csv")
    write(blur_kernel(), "blur_kernel.csv")
    write(jpeg_quantisation(), "jpeg_quantisation.csv")
    write(recompression_steps(), "recompression_steps.csv")
    write(palette_colours(), "palette_colours.csv")
