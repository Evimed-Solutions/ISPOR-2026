from __future__ import annotations

from runtime import run_root

import argparse
import io
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from PIL import Image, ImageFilter

from km_synthetic_utils import (
    ARM_CONTROL,
    ARM_TREATMENT,
    ARMS,
    build_km_outputs,
    rmst_from_step,
    simulate_trial,
    stable_rng,
    step_survival_at,
)


KNOB_COLUMNS = [
    "dpi",
    "jpeg_quality",
    "recompression_generations",
    "blur_sigma_px",
    "chroma",
    "colour_palette",
    "grayscale",
    "paper_noise_sd",
    "skew_degrees",
]
THRESHOLD_FACTORS = {"dpi", "jpeg_quality", "blur", "recompression"}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_command(args: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True, timeout=timeout)


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def ensure_dirs(root: Path) -> None:
    for rel_path in (
        "data/truth",
        "data/processed/kmgpt_outputs",
        "data/vector",
        "data/images",
        "data/run_images",
        "results/tables",
    ):
        (root / rel_path).mkdir(parents=True, exist_ok=True)




def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def empty_if_none(value: object) -> object:
    return "" if value is None else value


def build_image_conditions(cfg: dict) -> list[dict[str, object]]:
    defaults = dict(cfg["defaults"])
    fixed_render = {
        "risk_table": cfg["render"]["fixed_risk_table"],
        "censor_ticks": as_bool(cfg["render"]["fixed_censor_ticks"]),
        "annotations": cfg["render"]["fixed_annotations"],
    }
    conditions: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in cfg["images"]:
        row = {**defaults, **fixed_render, **entry}
        image_id = str(row["image_id"])
        if image_id in seen:
            raise ValueError(f"Duplicate image_id in v3 config: {image_id}")
        seen.add(image_id)
        for column in KNOB_COLUMNS:
            row[column] = empty_if_none(row.get(column, defaults.get(column, "")))
        row["dpi"] = int(row["dpi"])
        row["recompression_generations"] = int(row["recompression_generations"])
        row["blur_sigma_px"] = float(row["blur_sigma_px"])
        row["paper_noise_sd"] = float(row["paper_noise_sd"])
        row["skew_degrees"] = float(row["skew_degrees"])
        row["grayscale"] = as_bool(row["grayscale"])
        row["threshold_eligible"] = as_bool(row["threshold_eligible"])
        row["scenario"] = "" if row.get("scenario") is None else str(row.get("scenario", ""))
        row["jpeg_quality"] = "" if row.get("jpeg_quality") in (None, "") else int(row["jpeg_quality"])
        conditions.append(row)
    return conditions


def risk_table_times(cfg: dict, level: str) -> list[float]:
    if level == "sparse":
        return [float(x) for x in cfg["trial"]["sparse_risk_table_times_months"]]
    return [float(x) for x in cfg["trial"]["risk_table_times_months"]]


def render_pdf(root: Path, cfg: dict, trial: pd.DataFrame, km_curve: pd.DataFrame, condition: dict[str, object]) -> tuple[Path, list[float]]:
    render_cfg = cfg["render"]
    width = float(render_cfg["figure_width_inches"])
    height = float(render_cfg["figure_height_inches"])
    risk_level = str(condition["risk_table"])
    has_risk_table = risk_level != "absent"
    axis_rect = render_cfg["axis_rect"]["with_risk_table"] if has_risk_table else render_cfg["axis_rect"]["absent"]
    palette = render_cfg["colour_palettes"][str(condition["colour_palette"])]
    horizon = float(cfg["trial"]["admin_followup_months"])

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(axis_rect)
    for arm in ARMS:
        arm_df = km_curve[km_curve["arm"] == arm].sort_values("time_months")
        color = palette[arm]
        ax.step(
            arm_df["time_months"],
            arm_df["survival"],
            where="post",
            lw=float(render_cfg["line_width_points"]),
            color=color,
            label="Control" if arm == ARM_CONTROL else "Treatment",
        )
        if bool(condition["censor_ticks"]):
            cens = trial[(trial["arm"] == arm) & (trial["event"] == 0)]["observed_time_months"].to_numpy(float)
            cens = np.sort(cens[(cens >= 0) & (cens <= horizon)])
            if len(cens) > 45:
                cens = cens[np.linspace(0, len(cens) - 1, 45).astype(int)]
            surv = step_survival_at(arm_df, cens)
            ax.plot(
                cens,
                surv,
                linestyle="none",
                marker="+",
                markersize=float(render_cfg["censor_tick_size_points"]),
                markeredgewidth=0.9,
                color=color,
            )

    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(render_cfg["x_ticks"])
    ax.set_yticks(render_cfg["y_ticks"])
    ax.set_xlabel("Months")
    ax.set_ylabel("Overall survival")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=False)

    if str(condition["annotations"]) == "on_curve":
        ax.text(4.0, 0.18, "Log-rank p = 0.031", fontsize=8, color="#111111")

    if has_risk_table:
        font_size = float(render_cfg["risk_table_font_size"][risk_level])
        y0 = axis_rect[1] - (0.13 if risk_level == "clear" else 0.10)
        label_x = axis_rect[0] - 0.055
        fig.text(label_x, y0, "At risk", ha="right", va="center", fontsize=font_size)
        times = risk_table_times(cfg, risk_level)
        for row_idx, arm in enumerate(ARMS):
            label_y = y0 - (row_idx + 1) * 0.066
            fig.text(label_x, label_y, "Ctrl" if arm == ARM_CONTROL else "Trt", ha="right", va="center", fontsize=font_size)
            durations = trial.loc[trial["arm"] == arm, "observed_time_months"].to_numpy(float)
            for t in times:
                x_frac = axis_rect[0] + axis_rect[2] * t / horizon
                fig.text(x_frac, label_y, str(int(np.sum(durations >= t))), ha="center", va="center", fontsize=font_size)

    pdf_path = root / "data/vector" / f"{condition['image_id']}.pdf"
    fig.savefig(pdf_path, metadata={"Creator": "kmgpt_synthetic_image_quality_v3"})
    plt.close(fig)
    return pdf_path, [float(x) for x in axis_rect]


def rasterize_pdf(root: Path, pdf_path: Path, dpi: int) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="v3_raster_") as tmp_name:
        prefix = Path(tmp_name) / "render"
        run_command(["pdftoppm", "-png", "-singlefile", "-r", str(dpi), str(pdf_path), str(prefix)], root)
        image_path = prefix.with_suffix(".png")
        image = Image.open(image_path).convert("RGB")
        image.load()
        return image


def apply_chroma_420(image: Image.Image) -> Image.Image:
    y, cb, cr = image.convert("YCbCr").split()
    half = (max(1, image.width // 2), max(1, image.height // 2))
    cb_small = cb.resize(half, Image.Resampling.BICUBIC)
    cr_small = cr.resize(half, Image.Resampling.BICUBIC)
    cb_big = cb_small.resize(image.size, Image.Resampling.BICUBIC)
    cr_big = cr_small.resize(image.size, Image.Resampling.BICUBIC)
    return Image.merge("YCbCr", (y, cb_big, cr_big)).convert("RGB")


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), subsampling=0)
    buffer.seek(0)
    out = Image.open(buffer).convert("RGB")
    out.load()
    return out


def apply_recompression(image: Image.Image, generations: int, quality: int) -> Image.Image:
    out = image.convert("RGB")
    for _ in range(int(generations)):
        out = jpeg_roundtrip(out, int(quality))
    return out


def apply_legacy_scan(image: Image.Image, rng: np.random.Generator, paper_noise_sd: float, skew_degrees: float) -> Image.Image:
    out = image.convert("RGB")
    if paper_noise_sd > 0:
        arr = np.asarray(out).astype(float)
        paper = rng.normal(0.0, float(paper_noise_sd), size=arr.shape[:2])[:, :, None]
        texture = rng.normal(0.0, float(paper_noise_sd) / 2.0, size=arr.shape[:2])[:, :, None]
        arr = np.clip(arr + paper + texture, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr, "RGB")
    if abs(float(skew_degrees)) > 0:
        out = out.rotate(float(skew_degrees), resample=Image.Resampling.BICUBIC, fillcolor="white")
    return out


def degrade_image(root: Path, cfg: dict, image: Image.Image, condition: dict[str, object]) -> Image.Image:
    out = image.convert("RGB")
    blur_sigma = float(condition["blur_sigma_px"])
    if blur_sigma > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    if str(condition["chroma"]) == "420":
        out = apply_chroma_420(out)
    if condition["jpeg_quality"] != "":
        out = jpeg_roundtrip(out, int(condition["jpeg_quality"]))
    if int(condition["recompression_generations"]) > 0:
        out = apply_recompression(out, int(condition["recompression_generations"]), int(cfg["degradation"]["recompression_quality"]))
    if float(condition["paper_noise_sd"]) > 0 or abs(float(condition["skew_degrees"])) > 0:
        rng = stable_rng(cfg, condition["image_id"], "legacy_scan")
        out = apply_legacy_scan(out, rng, float(condition["paper_noise_sd"]), float(condition["skew_degrees"]))
    if bool(condition["grayscale"]):
        out = out.convert("L").convert("RGB")
    return out


def truth_metrics(cfg: dict, km_curve: pd.DataFrame) -> pd.DataFrame:
    tau = float(cfg["scoring"]["rmst_tau_months"])
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        arm_curve = km_curve[km_curve["arm"] == arm].sort_values("time_months")
        rows.append(
            {
                "arm": arm,
                "true_rmst_months": rmst_from_step(arm_curve, tau),
                "rmst_tau_months": tau,
            }
        )
    return pd.DataFrame(rows)


def render_images(root: Path, cfg: dict, trial: pd.DataFrame, km_curve: pd.DataFrame) -> pd.DataFrame:
    if not tool_available("pdftoppm"):
        raise RuntimeError("pdftoppm is required for v3 rasterisation but was not found on PATH.")

    rows: list[dict[str, object]] = []
    for condition in build_image_conditions(cfg):
        pdf_path, axis_rect = render_pdf(root, cfg, trial, km_curve, condition)
        raster = rasterize_pdf(root, pdf_path, int(condition["dpi"]))
        final_image = degrade_image(root, cfg, raster, condition)
        final_path = root / "data/images" / f"{condition['image_id']}.png"
        final_image.save(final_path, format="PNG")
        width, height = final_image.size
        rows.append(
            {
                "image_id": condition["image_id"],
                "image_path": rel(final_path, root),
                "analysis_block": condition["analysis_block"],
                "factor": condition["factor"],
                "factor_level": condition["factor_level"],
                "factor_type": condition["factor_type"],
                "threshold_eligible": bool(condition["threshold_eligible"]),
                "scenario": condition["scenario"],
                "dpi": int(condition["dpi"]),
                "jpeg_quality": condition["jpeg_quality"],
                "recompression_generations": int(condition["recompression_generations"]),
                "blur_sigma_px": float(condition["blur_sigma_px"]),
                "chroma": condition["chroma"],
                "colour_palette": condition["colour_palette"],
                "grayscale": bool(condition["grayscale"]),
                "paper_noise_sd": float(condition["paper_noise_sd"]),
                "skew_degrees": float(condition["skew_degrees"]),
                "risk_table": condition["risk_table"],
                "censor_ticks": bool(condition["censor_ticks"]),
                "annotations": condition["annotations"],
                "pdf_path": rel(pdf_path, root),
                "image_width": width,
                "image_height": height,
                "axis_left_frac": axis_rect[0],
                "axis_bottom_frac": axis_rect[1],
                "axis_width_frac": axis_rect[2],
                "axis_height_frac": axis_rect[3],
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "data/truth/image_manifest.csv", index=False)
    return manifest


def build_run_sheet(root: Path, cfg: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    by_image = {str(row["image_id"]): row for _, row in manifest.iterrows()}
    replicates_per_image = int(cfg.get("kmgpt", {}).get("replicates_per_image", 3))
    if replicates_per_image < 1:
        raise ValueError("kmgpt.replicates_per_image must be at least 1.")

    def run_image_path(run_id: str, source_rel_path: str) -> str:
        source_path = root / source_rel_path
        if not source_path.exists():
            raise FileNotFoundError(f"Run image source does not exist: {source_path}")
        run_dir = root / "data/run_images" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        destination = run_dir / f"{run_id}.png"
        shutil.copy2(source_path, destination)
        return rel(destination, root)

    def run_row(image: pd.Series, replicate_id: int) -> dict[str, object]:
        image_id = str(image["image_id"])
        run_id = f"{image_id}_rep{replicate_id:02d}"
        primary_run_id = f"{image_id}_rep01"
        source_image_path = str(image["image_path"])
        is_repeat = replicate_id > 1
        return {
            "run_id": run_id,
            "image_id": image_id,
            "replicate_id": replicate_id,
            "image_path": run_image_path(run_id, source_image_path),
            "source_image_path": source_image_path,
            "analysis_block": str(image["analysis_block"]),
            "factor": str(image["factor"]),
            "factor_level": image["factor_level"],
            "scenario": str(image["scenario"]),
            "is_repeat": is_repeat,
            "repeat_of": primary_run_id if is_repeat else "",
            "km_gpt_completed": False,
            "km_gpt_output_file": f"data/processed/kmgpt_outputs/{run_id}.csv",
            "notes": "Repeat KM-GPT run of same image." if is_repeat else "",
        }

    rows: list[dict[str, object]] = []

    def append_image_replicates(image_id: str) -> None:
        if image_id not in by_image:
            return
        image = by_image[image_id]
        for replicate_id in range(1, replicates_per_image + 1):
            rows.append(run_row(image, replicate_id))

    emitted_images: set[str] = set()
    for image_id in ("base_clean_300dpi", "workflow_scanned_legacy"):
        append_image_replicates(image_id)
        emitted_images.add(image_id)
    for _, image in manifest.iterrows():
        image_id = str(image["image_id"])
        if image_id in emitted_images:
            continue
        append_image_replicates(image_id)
        emitted_images.add(image_id)

    sheet = pd.DataFrame(rows)
    run_sheet_path = root / "data/truth/km_gpt_run_sheet.csv"
    try:
        sheet.to_csv(run_sheet_path, index=False)
    except PermissionError:
        fallback_path = root / "data/processed/km_gpt_run_sheet_grouped.csv"
        sheet.to_csv(fallback_path, index=False)
        print(f"Warning: could not overwrite locked run sheet; wrote grouped run sheet to {fallback_path}")
    return sheet


def values_differ(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return False
    return str(left) != str(right)


def validate_outputs(root: Path, cfg: dict, manifest: pd.DataFrame, run_sheet: pd.DataFrame) -> None:
    replicates_per_image = int(cfg.get("kmgpt", {}).get("replicates_per_image", 3))
    expected_run_rows = len(manifest) * replicates_per_image
    if len(manifest) != 25:
        raise AssertionError(f"Expected 25 unique images, found {len(manifest)}.")
    if manifest["image_id"].nunique() != 25:
        raise AssertionError("image_manifest.csv does not contain 25 unique image_id values.")
    if len(run_sheet) != expected_run_rows:
        raise AssertionError(f"Expected {expected_run_rows} KM-GPT run-sheet rows, found {len(run_sheet)}.")
    image_files = sorted((root / "data/images").glob("*.png"))
    if len(image_files) != 25:
        raise AssertionError(f"Expected 25 generated PNG files, found {len(image_files)}.")
    run_image_files = sorted((root / "data/run_images").glob("*/*.png"))
    if len(run_image_files) != expected_run_rows:
        raise AssertionError(f"Expected {expected_run_rows} run-specific PNG files, found {len(run_image_files)}.")
    run_counts = run_sheet.groupby("image_id")["run_id"].nunique()
    bad_run_counts = run_counts[run_counts != replicates_per_image]
    if not bad_run_counts.empty:
        raise AssertionError(f"Each image must have {replicates_per_image} KM-GPT runs; bad counts: {bad_run_counts.to_dict()}.")
    if manifest["image_path"].astype(str).str.contains("data/calibration|data/v2|Real KM", regex=True).any():
        raise AssertionError("v3 image manifest contains a non-v3 or real-figure path.")
    if run_sheet["image_path"].astype(str).str.contains("data/calibration|data/v2|Real KM", regex=True).any():
        raise AssertionError("v3 run sheet contains a non-v3 or real-figure path.")
    threshold_bad = manifest[
        (manifest["threshold_eligible"].astype(bool))
        & ~manifest["factor"].isin(sorted(THRESHOLD_FACTORS))
    ]
    if not threshold_bad.empty:
        raise AssertionError("Only DPI, JPEG, blur, and recompression may be threshold eligible.")

    defaults = cfg["defaults"]
    normalized_defaults = {
        "dpi": int(defaults["dpi"]),
        "jpeg_quality": "",
        "recompression_generations": int(defaults["recompression_generations"]),
        "blur_sigma_px": float(defaults["blur_sigma_px"]),
        "chroma": str(defaults["chroma"]),
        "colour_palette": str(defaults["colour_palette"]),
        "grayscale": as_bool(defaults["grayscale"]),
        "paper_noise_sd": float(defaults["paper_noise_sd"]),
        "skew_degrees": float(defaults["skew_degrees"]),
    }
    sweep_rows = manifest[manifest["analysis_block"] == "factor_sweep"]
    for _, row in sweep_rows.iterrows():
        changed = [
            knob
            for knob in KNOB_COLUMNS
            if values_differ(row[knob], normalized_defaults[knob])
        ]
        if len(changed) != 1:
            raise AssertionError(f"Factor-sweep row {row['image_id']} changed {changed}; expected exactly one knob.")
    jpeg_sweeps = sweep_rows[sweep_rows["factor"] == "jpeg_quality"]
    if not jpeg_sweeps.empty and (jpeg_sweeps["chroma"].astype(str) != "none").any():
        raise AssertionError("JPEG quality sweep rows must keep chroma=none.")
    repeats = run_sheet[run_sheet["is_repeat"].astype(bool)]
    run_ids = set(run_sheet["run_id"].astype(str))
    for _, row in repeats.iterrows():
        source = manifest[manifest["image_id"] == row["image_id"]]
        if source.empty or str(source.iloc[0]["image_path"]) != str(row["source_image_path"]):
            raise AssertionError(f"Repeat row {row['run_id']} does not reuse the source image content.")
        if str(row["repeat_of"]) not in run_ids:
            raise AssertionError(f"Repeat row {row['run_id']} references missing primary run {row['repeat_of']}.")
        if not (root / str(row["image_path"])).exists():
            raise AssertionError(f"Repeat row {row['run_id']} points to a missing run-specific image.")


def write_metadata(root: Path, cfg: dict) -> None:
    metadata = {
        "seed": cfg["seed"],
        "protocol": "km_gpt_synthetic_image_quality_v3",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
        "pdftoppm_available": tool_available("pdftoppm"),
    }
    with (root / "data/truth/run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def write_generation_summary(root: Path, manifest: pd.DataFrame, run_sheet: pd.DataFrame, truth_by_arm: pd.DataFrame) -> None:
    factor_counts = manifest["factor"].value_counts().to_dict()
    scenario_rows = manifest[manifest["analysis_block"].isin(["base_reference", "workflow_scenario"])]
    replicates_per_image = int(run_sheet["replicate_id"].max()) if "replicate_id" in run_sheet.columns else 1
    lines = [
        "# v3 KM-GPT Synthetic Image-Quality Generation Summary",
        "",
        "Active workflow: fully synthetic, KM-GPT-only RMST image-quality study.",
        "",
        f"- Unique synthetic images: {len(manifest)}.",
        f"- KM-GPT run-sheet rows: {len(run_sheet)}.",
        f"- KM-GPT runs per image: {replicates_per_image}.",
        "- KM-GPT upload images are grouped one folder per run under `data/run_images/<run_id>/`.",
        f"- Factor counts: {factor_counts}.",
        "",
        "## RMST Truth",
        "",
    ]
    for _, row in truth_by_arm.iterrows():
        lines.append(
            f"- {row['arm']}: RMST({row['rmst_tau_months']:.0f}) {row['true_rmst_months']:.2f} months."
        )
    lines.extend(
        [
            "",
            "## Scenario Rows",
            "",
        ]
    )
    for _, row in scenario_rows.iterrows():
        lines.append(f"- {row['image_id']}: scenario `{row['scenario']}`; image `{row['image_path']}`.")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Upload images according to `data/truth/km_gpt_run_sheet.csv`, standardise KM-GPT curve points in `data/processed/kmgpt_curve_points.csv.gz`, then run `scripts/core/score_rmst.py`.",
        ]
    )
    (root / "results/tables/v3_generation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v3 synthetic KM images and KM-GPT run sheet.")
    parser.add_argument("--config", type=Path, default=Path("config/study.yaml"))
    args = parser.parse_args()
    root = run_root()
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_config(cfg_path)
    if (root / "data/images").exists() and any((root / "data/images").glob("*.png")):
        raise FileExistsError("Generation destination already contains images; choose a fresh --output-dir")
    ensure_dirs(root)

    rng = np.random.default_rng(int(cfg["seed"]))
    trial = simulate_trial(cfg, rng)
    km_curve, km_grid, risk_table = build_km_outputs(trial, cfg)
    truth_by_arm = truth_metrics(cfg, km_curve)

    trial.to_csv(root / "data/truth/base_trial.csv", index=False)
    km_curve.to_csv(root / "data/truth/truth_km_curve.csv", index=False)
    km_grid.to_csv(root / "data/truth/truth_km_grid.csv", index=False)
    risk_table.to_csv(root / "data/truth/truth_risk_table.csv", index=False)
    truth_by_arm.to_csv(root / "data/truth/truth_summary_by_arm.csv", index=False)

    manifest = render_images(root, cfg, trial, km_curve)
    run_sheet = build_run_sheet(root, cfg, manifest)
    validate_outputs(root, cfg, manifest, run_sheet)
    write_metadata(root, cfg)
    write_generation_summary(root, manifest, run_sheet, truth_by_arm)

    print(f"Generated {len(manifest)} unique v3 images.")
    print(f"Wrote {len(run_sheet)} KM-GPT run-sheet rows.")
    print(root / "data/truth/image_manifest.csv")
    print(root / "data/truth/km_gpt_run_sheet.csv")


if __name__ == "__main__":
    main()
