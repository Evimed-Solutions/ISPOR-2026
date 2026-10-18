from __future__ import annotations

from runtime import run_root

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from km_synthetic_utils import ARM_CONTROL, ARM_TREATMENT, ARMS, rmst_from_step, step_survival_at


CURVE_COLUMNS = [
    "run_id",
    "image_id",
    "extracted_arm_label",
    "time_months",
    "survival",
    "notes",
    "raw_curve_id",
    "raw_curve_label",
    "manifest_annotation_label",
    "manifest_annotation_source",
    "import_status",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def numeric(value: object) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def ensure_dirs(root: Path) -> None:
    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "results/tables").mkdir(parents=True, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write the requested output; a locked file must fail visibly."""
    df.to_csv(path, index=False)
    return path


def create_curve_template(root: Path) -> Path:
    run_sheet_path = root / "data/truth/km_gpt_run_sheet.csv"
    if not run_sheet_path.exists():
        raise FileNotFoundError("Run v3 image generation first; km_gpt_run_sheet.csv is missing.")
    run_sheet = pd.read_csv(run_sheet_path, keep_default_na=False)
    rows = [
        {
            "run_id": row["run_id"],
            "image_id": row["image_id"],
            "extracted_arm_label": "",
            "time_months": "",
            "survival": "",
            "notes": "Add one row per extracted curve point; survival must be a 0-1 fraction.",
        }
        for _, row in run_sheet.iterrows()
    ]
    template = root / "data/processed/kmgpt_curve_points_template.csv"
    pd.DataFrame(rows, columns=CURVE_COLUMNS).to_csv(template, index=False)
    return template


def load_curve_points(path: Path) -> pd.DataFrame:
    points = pd.read_csv(path, keep_default_na=False)
    for column in CURVE_COLUMNS:
        if column not in points.columns:
            points[column] = ""
    points = points[CURVE_COLUMNS].copy()
    points["time_months"] = points["time_months"].map(numeric)
    points["survival"] = points["survival"].map(numeric)
    return points


def prepare_step_curve(points: pd.DataFrame) -> pd.DataFrame:
    curve = points[["time_months", "survival"]].dropna().copy()
    curve = curve[np.isfinite(curve["time_months"]) & np.isfinite(curve["survival"])]
    curve = curve[(curve["survival"] >= -0.02) & (curve["survival"] <= 1.02)]
    if curve.empty:
        return pd.DataFrame(columns=["time_months", "survival"])
    curve["survival"] = curve["survival"].clip(0.0, 1.0)
    curve = (
        curve.groupby("time_months", as_index=False)["survival"]
        .min()
        .sort_values("time_months")
        .reset_index(drop=True)
    )
    first_time = float(curve.iloc[0]["time_months"])
    if first_time > 0:
        curve = pd.concat(
            [
                pd.DataFrame([{"time_months": 0.0, "survival": float(curve.iloc[0]["survival"])}]),
                curve,
            ],
            ignore_index=True,
        )
    return curve.sort_values("time_months").reset_index(drop=True)


def axis_is_usable(curves: dict[str, pd.DataFrame], tau: float) -> tuple[bool, str]:
    all_times = pd.concat([curve["time_months"] for curve in curves.values()], ignore_index=True).to_numpy(float)
    all_times = all_times[np.isfinite(all_times)]
    if len(all_times) == 0:
        return False, "no_finite_times"
    min_time = float(np.min(all_times))
    max_time = float(np.max(all_times))
    if min_time < -0.5 or min_time > 3.0:
        return False, f"time_axis_min_out_of_range:{min_time:.3g}"
    if max_time < 0.75 * tau or max_time > 1.5 * tau:
        return False, f"time_axis_max_out_of_range:{max_time:.3g}"
    return True, ""


def truth_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    truth_summary = pd.read_csv(root / "data/truth/truth_summary_by_arm.csv", keep_default_na=False)
    truth_grid = pd.read_csv(root / "data/truth/truth_km_grid.csv", keep_default_na=False)
    truth_curves = {
        arm: truth_grid[truth_grid["arm"] == arm][["time_months", "survival"]].astype(float).sort_values("time_months")
        for arm in ARMS
    }
    return truth_summary, truth_grid, truth_curves


def match_arms_by_shape(
    extracted_curves: dict[str, pd.DataFrame],
    truth_curves: dict[str, pd.DataFrame],
    grid: np.ndarray,
) -> tuple[dict[str, str], dict[str, float], float]:
    labels = list(extracted_curves)
    best_mapping: dict[str, str] = {}
    best_rmse_by_label: dict[str, float] = {}
    best_total = float("inf")
    for truth_order in permutations(ARMS, len(labels)):
        total = 0.0
        rmse_by_label: dict[str, float] = {}
        mapping = dict(zip(labels, truth_order))
        for label, arm in mapping.items():
            extracted_surv = step_survival_at(extracted_curves[label], grid)
            truth_surv = step_survival_at(truth_curves[arm], grid)
            rmse = float(np.sqrt(np.mean((extracted_surv - truth_surv) ** 2)))
            rmse_by_label[label] = rmse
            total += rmse
        if total < best_total:
            best_total = total
            best_mapping = mapping
            best_rmse_by_label = rmse_by_label
    return best_mapping, best_rmse_by_label, best_total


def run_metadata(root: Path) -> pd.DataFrame:
    run_sheet = pd.read_csv(root / "data/truth/km_gpt_run_sheet.csv", keep_default_na=False)
    manifest = pd.read_csv(root / "data/truth/image_manifest.csv", keep_default_na=False)
    manifest_cols = [
        "image_id",
        "analysis_block",
        "factor",
        "factor_level",
        "factor_type",
        "threshold_eligible",
        "scenario",
    ]
    return run_sheet.merge(manifest[manifest_cols], on="image_id", how="left", suffixes=("", "_manifest"))


def import_status_by_run(root: Path) -> pd.DataFrame:
    status_path = root / "data/processed/kmgpt_import_run_status.csv"
    if not status_path.exists():
        return pd.DataFrame(columns=["run_id", "import_status", "failure_reason"])
    status = pd.read_csv(status_path, keep_default_na=False)
    keep = [column for column in ["run_id", "import_status", "failure_reason"] if column in status.columns]
    return status[keep].drop_duplicates("run_id")


def not_scored_run(row: pd.Series, reason: str) -> dict[str, object]:
    return {
        "run_id": row["run_id"],
        "image_id": row["image_id"],
        "replicate_id": row.get("replicate_id", ""),
        "analysis_block": row.get("analysis_block", ""),
        "factor": row.get("factor", ""),
        "factor_level": row.get("factor_level", ""),
        "factor_type": row.get("factor_type", ""),
        "threshold_eligible": row.get("threshold_eligible", ""),
        "scenario": row.get("scenario", ""),
        "scoring_status": "not_scored",
        "status_reason": reason,
        "rmst_tau_months": np.nan,
        "control_rmst_error_months": np.nan,
        "treatment_rmst_error_months": np.nan,
        "control_abs_rmst_error_months": np.nan,
        "treatment_abs_rmst_error_months": np.nan,
        "mean_abs_arm_rmst_error_months": np.nan,
        "rmst_difference_error_months": np.nan,
        "shape_rmse_total": np.nan,
    }


def score_curve_points(root: Path, cfg: dict, points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tau = float(cfg["scoring"]["rmst_tau_months"])
    truth_summary, _truth_grid, truth_curves = truth_frames(root)
    truth_by_arm = {str(row["arm"]): float(row["true_rmst_months"]) for _, row in truth_summary.iterrows()}
    true_diff = truth_by_arm[ARM_TREATMENT] - truth_by_arm[ARM_CONTROL]
    common_grid = np.arange(0.0, tau + 1e-9, float(cfg["extraction"]["grid_step_months"]))
    meta = run_metadata(root)
    import_status = import_status_by_run(root)
    if not import_status.empty:
        meta = meta.merge(import_status, on="run_id", how="left")

    arm_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for _, run in meta.iterrows():
        run_id = str(run["run_id"])
        status = str(run.get("import_status", "")).strip()
        if status and status != "accepted":
            reason = str(run.get("failure_reason", "")).strip() or status
            run_rows.append(not_scored_run(run, reason))
            continue
        run_points = points[points["run_id"].astype(str) == run_id]
        curves: dict[str, pd.DataFrame] = {}
        for label, group in run_points.groupby("extracted_arm_label", sort=False):
            label = str(label).strip()
            if label == "":
                continue
            curve = prepare_step_curve(group)
            if len(curve) >= 2:
                curves[label] = curve
        if len(curves) != 2:
            run_rows.append(not_scored_run(run, f"usable_arm_count:{len(curves)}"))
            continue
        usable_axis, reason = axis_is_usable(curves, tau)
        if not usable_axis:
            run_rows.append(not_scored_run(run, reason))
            continue

        mapping, rmse_by_label, rmse_total = match_arms_by_shape(curves, truth_curves, common_grid)
        extracted_rmst = {label: rmst_from_step(curve, tau) for label, curve in curves.items()}
        assigned_rmst: dict[str, float] = {}
        for label, arm in mapping.items():
            assigned_rmst[arm] = extracted_rmst[label]
            error = extracted_rmst[label] - truth_by_arm[arm]
            arm_rows.append(
                {
                    "run_id": run_id,
                    "image_id": run["image_id"],
                    "replicate_id": run.get("replicate_id", ""),
                    "extracted_arm_label": label,
                    "assigned_arm": arm,
                    "scoring_status": "scored",
                    "rmst_tau_months": tau,
                    "extracted_rmst_months": extracted_rmst[label],
                    "true_rmst_months": truth_by_arm[arm],
                    "rmst_error_months": error,
                    "abs_rmst_error_months": abs(error),
                    "shape_rmse": rmse_by_label[label],
                }
            )
        control_error = assigned_rmst[ARM_CONTROL] - truth_by_arm[ARM_CONTROL]
        treatment_error = assigned_rmst[ARM_TREATMENT] - truth_by_arm[ARM_TREATMENT]
        run_rows.append(
            {
                "run_id": run_id,
                "image_id": run["image_id"],
                "replicate_id": run.get("replicate_id", ""),
                "analysis_block": run.get("analysis_block", ""),
                "factor": run.get("factor", ""),
                "factor_level": run.get("factor_level", ""),
                "factor_type": run.get("factor_type", ""),
                "threshold_eligible": run.get("threshold_eligible", ""),
                "scenario": run.get("scenario", ""),
                "scoring_status": "scored",
                "status_reason": "",
                "rmst_tau_months": tau,
                "control_rmst_error_months": control_error,
                "treatment_rmst_error_months": treatment_error,
                "control_abs_rmst_error_months": abs(control_error),
                "treatment_abs_rmst_error_months": abs(treatment_error),
                "mean_abs_arm_rmst_error_months": float(np.mean([abs(control_error), abs(treatment_error)])),
                "rmst_difference_error_months": (assigned_rmst[ARM_TREATMENT] - assigned_rmst[ARM_CONTROL]) - true_diff,
                "shape_rmse_total": rmse_total,
            }
        )

    arm_scores = pd.DataFrame(arm_rows)
    run_scores = pd.DataFrame(run_rows)
    condition_scores = condition_summary(run_scores)
    return arm_scores, run_scores, condition_scores


def summarize(series: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {f"{prefix}_mean": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def condition_summary(run_scores: pd.DataFrame) -> pd.DataFrame:
    if run_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["image_id", "analysis_block", "factor", "factor_level", "factor_type", "threshold_eligible", "scenario"]
    for key, group in run_scores.groupby(group_cols, dropna=False, sort=False):
        scored = group[group["scoring_status"] == "scored"]
        row = dict(zip(group_cols, key))
        row.update(
            {
                "total_runs": int(len(group)),
                "scored_runs": int(len(scored)),
                "not_scored_runs": int((group["scoring_status"] != "scored").sum()),
            }
        )
        for column in (
            "control_rmst_error_months",
            "treatment_rmst_error_months",
            "mean_abs_arm_rmst_error_months",
            "rmst_difference_error_months",
        ):
            row.update(summarize(scored[column], column))
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(root: Path, input_path: Path, run_scores: pd.DataFrame | None, condition_scores: pd.DataFrame | None) -> None:
    lines = [
        "# v3 KM-GPT RMST Results Summary",
        "",
        "Active workflow: fully synthetic KM-GPT-only RMST image-quality study.",
        "",
    ]
    if run_scores is None or condition_scores is None:
        lines.extend(
            [
                "KM-GPT curve points have not been scored yet.",
                "",
                f"Fill `data/processed/kmgpt_curve_points.csv.gz` using the template generated from `{input_path.name}`, then rerun this script.",
            ]
        )
    else:
        scored_runs = int((run_scores["scoring_status"] == "scored").sum())
        not_scored_runs = int((run_scores["scoring_status"] != "scored").sum())
        failure_counts = (
            run_scores.loc[run_scores["scoring_status"] != "scored", "status_reason"]
            .astype(str)
            .replace("", "not_scored")
            .value_counts()
        )
        failure_lines = [f"- {reason}: {count} runs." for reason, count in failure_counts.items()]
        lines.extend(
            [
                f"- Scored KM-GPT runs: {scored_runs}.",
                f"- Not-scored KM-GPT runs: {not_scored_runs}.",
                f"- Conditions with at least one scored run: {int((condition_scores['scored_runs'] > 0).sum())}.",
                "",
                "## Non-Scored Run Reasons",
                "",
                *(failure_lines or ["- None."]),
                "",
                "## RMST Error Ranges",
                "",
                f"- Max absolute arm-level RMST error: {run_scores[['control_abs_rmst_error_months', 'treatment_abs_rmst_error_months']].max(skipna=True).max(skipna=True):.3f} months.",
                f"- Max absolute RMST-difference error: {run_scores['rmst_difference_error_months'].abs().max(skipna=True):.3f} months.",
                "",
                "Condition summaries report mean, min, and max RMST error across the three KM-GPT runs per image.",
            ]
        )
    (root / "results/tables/poster_results_summary_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score KM-GPT v3 curve points against synthetic RMST truth.")
    parser.add_argument("--config", type=Path, default=Path("config/study.yaml"))
    parser.add_argument("--curve-points", type=Path, default=Path("data/processed/kmgpt_curve_points.csv.gz"))
    args = parser.parse_args()

    root = run_root()
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_config(cfg_path)
    ensure_dirs(root)

    curve_path = args.curve_points if args.curve_points.is_absolute() else root / args.curve_points
    if not curve_path.exists():
        template = create_curve_template(root)
        write_summary(root, template, None, None)
        print(f"KM-GPT curve point file not found: {curve_path}")
        print(f"Wrote template: {template}")
        print(root / "results/tables/poster_results_summary_v3.md")
        return

    points = load_curve_points(curve_path)
    arm_scores, run_scores, condition_scores = score_curve_points(root, cfg, points)
    safe_to_csv(arm_scores, root / "data/processed/kmgpt_rmst_by_arm.csv")
    safe_to_csv(run_scores, root / "results/tables/kmgpt_rmst_error_scores.csv")
    safe_to_csv(condition_scores, root / "results/tables/kmgpt_rmst_error_by_condition.csv")
    write_summary(root, curve_path, run_scores, condition_scores)
    print(f"Scored {(run_scores['scoring_status'] == 'scored').sum()} KM-GPT runs.")
    print(root / "results/tables/kmgpt_rmst_error_scores.csv")
    print(root / "results/tables/kmgpt_rmst_error_by_condition.csv")
    print(root / "results/tables/poster_results_summary_v3.md")


if __name__ == "__main__":
    main()
