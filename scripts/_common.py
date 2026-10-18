"""Shared study geometry, input paths, and diagnostic table helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from runtime import PACKAGE_ROOT, run_root

HERE = PACKAGE_ROOT / "scripts"
ROOT = run_root()
FOLDER = ROOT / "results"
OUT = FOLDER / "diagnostics"
WORK = ROOT / "intermediates"
IMAGES = ROOT / "data/images"
RAW = ROOT / "data/processed"
TRUTH = ROOT / "data/truth"
RESULTS = ROOT / "results/tables"
ZIPS = sorted((ROOT / "data/extractor").glob("*.zip"))

TAU = 36.0
# Figure geometry from config/study.yaml (no risk table)
FIG_W_IN, FIG_H_IN = 6.4, 4.8
AXIS_LEFT, AXIS_BOTTOM, AXIS_WIDTH, AXIS_HEIGHT = 0.14, 0.16, 0.80, 0.73
Y_TOP = 1.02  # ylim upper bound used by the renderer
LINE_WIDTH_PT = 2.4
CENSOR_STROKE_PT = 0.9

FACTOR_ORDER = [
    "base_clean_300dpi", "dpi_200", "dpi_150", "dpi_120", "dpi_90", "dpi_75",
    "blur_0_3", "blur_0_7", "blur_1_1", "blur_1_5",
    "jpeg_95", "jpeg_85", "jpeg_75", "jpeg_65", "jpeg_55",
    "recompression_1", "recompression_3", "recompression_5",
    "chroma_420", "colour_easy", "colour_hard", "colour_grayscale",
    "workflow_publisher_raster_export", "workflow_screenshot_slide", "workflow_scanned_legacy",
]
CLEAN_REFERENCE = {300: "base_clean_300dpi", 200: "dpi_200", 150: "dpi_150", 120: "dpi_120", 90: "dpi_90", 75: "dpi_75"}




def load_scorer():
    """Import the study's own scoring module (scripts/core/score_rmst.py)."""
    spec = importlib.util.spec_from_file_location("score_v3", PACKAGE_ROOT / "scripts/core/score_rmst.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest() -> pd.DataFrame:
    return pd.read_csv(TRUTH / "image_manifest.csv").set_index("image_id")


def true_rmst() -> dict[str, float]:
    summary = pd.read_csv(TRUTH / "truth_summary_by_arm.csv")
    return dict(zip(summary["arm"], summary["true_rmst_months"].astype(float)))


def px_per_survival(dpi: float) -> float:
    return AXIS_HEIGHT * FIG_H_IN * dpi / Y_TOP


def px_per_month(dpi: float) -> float:
    return AXIS_WIDTH * FIG_W_IN * dpi / TAU


def safe_name(image_id: str) -> str:
    """SurvdigitizeR picks its image reader from the file name, so 'jpeg' in a PNG name breaks it."""
    return image_id.replace("jpeg", "jq")


def write(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)} ({len(df)} rows)")
    return path
