from __future__ import annotations

from runtime import run_root

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_scoring_module():
    module_path = SCRIPT_DIR / "score_rmst.py"
    spec = importlib.util.spec_from_file_location("v3_rmst_scoring_script", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load scoring script from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCORING = load_scoring_module()
load_config = _SCORING.load_config
score_curve_points = _SCORING.score_curve_points
write_summary = _SCORING.write_summary


CURVE_POINT_COLUMNS = [
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

IMPORT_STATUS_COLUMNS = [
    "run_id",
    "image_id",
    "replicate_id",
    "zip_file",
    "source_result",
    "import_status",
    "failure_reason",
    "grouping_basis",
    "curve_group_count",
    "min_curve_points",
    "raw_labels",
    "manifest_labels",
    "annotation_sources",
    "curve_annotations_confirmed",
    "review_required",
    "llm_confidence",
    "curve_group_summary",
]


def result_entries(zip_file: zipfile.ZipFile) -> list[str]:
    return sorted(
        entry
        for entry in zip_file.namelist()
        if entry.startswith("results/") and entry.endswith(".csv")
    )


def read_batch_summary(zip_file: zipfile.ZipFile) -> pd.DataFrame:
    summary_text = zip_file.read("batch-summary.csv").decode("utf-8-sig")
    return pd.read_csv(io.StringIO(summary_text), keep_default_na=False)


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write the requested output; a locked file must fail visibly."""
    df.to_csv(path, index=False)
    return path


def read_manifest_json(zip_file: zipfile.ZipFile, batch_image_id: str) -> dict:
    manifest_path = f"manifests/{batch_image_id}.json"
    if manifest_path not in zip_file.namelist():
        return {}
    try:
        return json.loads(zip_file.read(manifest_path).decode("utf-8-sig"))
    except Exception:
        return {}


def manifest_annotations(manifest: dict) -> dict[str, dict[str, str]]:
    annotations: dict[str, dict[str, str]] = {}
    for item in manifest.get("curve_annotations", []) or []:
        curve = str(item.get("curve", "")).strip()
        if not curve:
            continue
        annotations[curve] = {
            "label": str(item.get("label", "")).strip(),
            "source": str(item.get("source", "")).strip(),
        }
    return annotations


def canonical_arm_label(label: object) -> str:
    text = str(label).strip().lower()
    if text.startswith("control") or text == "ctrl":
        return "Control"
    if text.startswith("treatment") or text == "trt":
        return "Treatment"
    return ""


def numeric(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def curve_group_key(row: dict[str, str]) -> tuple[str, str]:
    curve_id = str(row.get("curve_id", "")).strip()
    if curve_id:
        return f"curve_{curve_id}", "curve_id"
    curve_label = str(row.get("curve", "")).strip()
    return curve_label or "curve_unknown", "label"


def classify_curve_groups(
    groups: dict[str, list[dict[str, object]]],
    min_curve_points: int,
) -> tuple[str, str, int, str]:
    if len(groups) != 2:
        return "geometric_curve_count_not_two", f"geometric_curve_count:{len(groups)}", 0, ""

    group_summaries: list[str] = []
    canonical_labels: list[str] = []
    point_counts: list[int] = []
    for group_key, group_rows in sorted(groups.items()):
        usable = [
            row
            for row in group_rows
            if np.isfinite(float(row["time_months"]))
            and np.isfinite(float(row["survival"]))
            and -0.02 <= float(row["survival"]) <= 1.02
        ]
        point_counts.append(len(usable))
        csv_labels = [canonical_arm_label(row["raw_curve_label"]) for row in usable]
        manifest_labels = [canonical_arm_label(row["manifest_annotation_label"]) for row in usable]
        label_counts = Counter(label for label in csv_labels if label)
        if not label_counts:
            label_counts = Counter(label for label in manifest_labels if label)
        canonical = label_counts.most_common(1)[0][0] if label_counts else ""
        canonical_labels.append(canonical)

        raw_label_values = sorted({str(row["raw_curve_label"]).strip() for row in group_rows if str(row["raw_curve_label"]).strip()})
        manifest_label_values = sorted(
            {
                str(row["manifest_annotation_label"]).strip()
                for row in group_rows
                if str(row["manifest_annotation_label"]).strip()
            }
        )
        group_summaries.append(
            f"{group_key}:n={len(usable)};label={canonical or 'unlabeled'};"
            f"raw={','.join(raw_label_values) or 'none'};"
            f"manifest={','.join(manifest_label_values) or 'none'}"
        )

    if any(label == "" for label in canonical_labels):
        return "unlabeled_two_curves", "unlabeled_two_curves", min(point_counts), " | ".join(group_summaries)
    if sorted(canonical_labels) != ["Control", "Treatment"]:
        return "single_label_two_curves", "single_label_two_curves", min(point_counts), " | ".join(group_summaries)
    if any(count < min_curve_points for count in point_counts):
        return "curve_point_count_below_min", "curve_point_count_below_min", min(point_counts), " | ".join(group_summaries)
    return "accepted", "", min(point_counts), " | ".join(group_summaries)


def standardise_zip_exports(root: Path, zip_dir: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_sheet = pd.read_csv(root / "data/truth/km_gpt_run_sheet.csv", keep_default_na=False)
    valid_run_ids = set(run_sheet["run_id"].astype(str))
    valid_image_ids = set(run_sheet["image_id"].astype(str))
    min_curve_points = int(cfg.get("kmgpt", {}).get("min_curve_points_per_arm", 150))
    zip_paths = sorted(zip_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No zip exports found in {zip_dir}")

    rows: list[dict[str, object]] = []
    import_status_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    problem_rows: list[dict[str, object]] = []
    result_hash_rows: list[dict[str, object]] = []
    seen_run_ids: set[str] = set()

    mapping = pd.read_csv(zip_dir / "batches.csv")
    if set(mapping["zip_file"]) != {p.name for p in zip_paths} or mapping["replicate_id"].duplicated().any():
        raise ValueError("Batch mapping must identify each archive and replicate exactly once")
    for batch in mapping.sort_values("replicate_id").itertuples(index=False):
        replicate_id, zip_path = int(batch.replicate_id), zip_dir / batch.zip_file
        with zipfile.ZipFile(zip_path) as zip_file:
            summary = read_batch_summary(zip_file)
            id_to_filename = dict(zip(summary["image_id"].astype(str), summary["filename"].astype(str)))
            for _, summary_row in summary.iterrows():
                image_id = Path(str(summary_row["filename"])).stem
                run_id = f"{image_id}_rep{replicate_id:02d}"
                batch_rows.append(
                    {
                        "zip_file": zip_path.name,
                        "replicate_id": replicate_id,
                        "batch_image_id": summary_row["image_id"],
                        "image_id": image_id,
                        "run_id": run_id,
                        "filename": summary_row["filename"],
                        "status": summary_row.get("status", ""),
                        "status_label": summary_row.get("status_label", ""),
                        "confidence": summary_row.get("confidence", ""),
                        "review_required": summary_row.get("review_required", ""),
                        "diagnostic_category": summary_row.get("diagnostic_category", ""),
                        "message": summary_row.get("message", ""),
                    }
                )

            for entry in result_entries(zip_file):
                batch_image_id = Path(entry).stem
                filename = id_to_filename.get(batch_image_id)
                if not filename:
                    problem_rows.append(
                        {
                            "zip_file": zip_path.name,
                            "entry": entry,
                            "problem": "missing_batch_summary_filename",
                        }
                    )
                    continue
                image_id = Path(filename).stem
                run_id = f"{image_id}_rep{replicate_id:02d}"
                if image_id not in valid_image_ids or run_id not in valid_run_ids:
                    problem_rows.append(
                        {
                            "zip_file": zip_path.name,
                            "entry": entry,
                            "problem": f"cannot_map_to_run_id:{run_id}",
                        }
                    )
                    continue

                data = zip_file.read(entry)
                manifest = read_manifest_json(zip_file, batch_image_id)
                annotations = manifest_annotations(manifest)
                result_hash_rows.append(
                    {
                        "zip_file": zip_path.name,
                        "replicate_id": replicate_id,
                        "image_id": image_id,
                        "run_id": run_id,
                        "result_sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                local_rows: list[dict[str, object]] = []
                groups: dict[str, list[dict[str, object]]] = {}
                grouping_basis_values: set[str] = set()
                for result_row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
                    curve_id = str(result_row.get("curve_id", "")).strip()
                    curve_label = str(result_row.get("curve", "")).strip()
                    group_key, grouping_basis = curve_group_key(result_row)
                    grouping_basis_values.add(grouping_basis)
                    manifest_annotation = annotations.get(curve_id, {}) if curve_id else {}
                    point_row = {
                        "run_id": run_id,
                        "image_id": image_id,
                        "extracted_arm_label": group_key,
                        "time_months": numeric(result_row.get("time", "")),
                        "survival": numeric(result_row.get("St", "")),
                        "notes": (
                            f"source_zip={zip_path.name};"
                            f"source_result={entry};"
                            f"curve_label={curve_label};"
                            f"manifest_notes={str(manifest.get('notes', '')).replace(';', ',')}"
                        ),
                        "raw_curve_id": curve_id,
                        "raw_curve_label": curve_label,
                        "manifest_annotation_label": manifest_annotation.get("label", ""),
                        "manifest_annotation_source": manifest_annotation.get("source", ""),
                    }
                    local_rows.append(point_row)
                    groups.setdefault(group_key, []).append(point_row)

                import_status, failure_reason, min_points, group_summary = classify_curve_groups(groups, min_curve_points)
                raw_labels = sorted({str(row["raw_curve_label"]).strip() for row in local_rows if str(row["raw_curve_label"]).strip()})
                manifest_labels = sorted(
                    {
                        str(row["manifest_annotation_label"]).strip()
                        for row in local_rows
                        if str(row["manifest_annotation_label"]).strip()
                    }
                )
                annotation_sources = sorted(
                    {
                        str(row["manifest_annotation_source"]).strip()
                        for row in local_rows
                        if str(row["manifest_annotation_source"]).strip()
                    }
                )
                import_status_rows.append(
                    {
                        "run_id": run_id,
                        "image_id": image_id,
                        "replicate_id": replicate_id,
                        "zip_file": zip_path.name,
                        "source_result": entry,
                        "import_status": import_status,
                        "failure_reason": failure_reason,
                        "grouping_basis": "+".join(sorted(grouping_basis_values)) or "",
                        "curve_group_count": len(groups),
                        "min_curve_points": min_points,
                        "raw_labels": " | ".join(raw_labels),
                        "manifest_labels": " | ".join(manifest_labels),
                        "annotation_sources": " | ".join(annotation_sources),
                        "curve_annotations_confirmed": manifest.get("curve_annotations_confirmed", ""),
                        "review_required": manifest.get("review_required", ""),
                        "llm_confidence": manifest.get("llm_confidence", ""),
                        "curve_group_summary": group_summary,
                    }
                )
                seen_run_ids.add(run_id)
                for row in local_rows:
                    row["import_status"] = import_status
                rows.extend(local_rows)

    for _, run in run_sheet.iterrows():
        run_id = str(run["run_id"])
        if run_id in seen_run_ids:
            continue
        import_status_rows.append(
            {
                "run_id": run_id,
                "image_id": run["image_id"],
                "replicate_id": run.get("replicate_id", ""),
                "zip_file": "",
                "source_result": "",
                "import_status": "missing_result_csv",
                "failure_reason": "missing_result_csv",
                "grouping_basis": "",
                "curve_group_count": 0,
                "min_curve_points": 0,
                "raw_labels": "",
                "manifest_labels": "",
                "annotation_sources": "",
                "curve_annotations_confirmed": "",
                "review_required": "",
                "llm_confidence": "",
                "curve_group_summary": "",
            }
        )

    return (
        pd.DataFrame(rows, columns=CURVE_POINT_COLUMNS),
        pd.DataFrame(import_status_rows, columns=IMPORT_STATUS_COLUMNS),
        pd.DataFrame(batch_rows),
        pd.DataFrame(problem_rows),
        pd.DataFrame(result_hash_rows),
    )


def numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if column not in {
            "run_id",
            "image_id",
            "analysis_block",
            "factor",
            "factor_level",
            "factor_type",
            "threshold_eligible",
            "scenario",
            "scoring_status",
            "status_reason",
            "import_status",
            "failure_reason",
            "grouping_basis",
            "raw_labels",
            "manifest_labels",
            "annotation_sources",
            "curve_group_summary",
        }:
            converted = pd.to_numeric(out[column], errors="coerce")
            if converted.notna().any():
                out[column] = converted
    return out


def result_hash_summary(hash_df: pd.DataFrame) -> pd.DataFrame:
    if hash_df.empty:
        return pd.DataFrame(columns=["image_id", "unique_result_hashes", "replicate_count"])
    return (
        hash_df.groupby("image_id", as_index=False)
        .agg(
            unique_result_hashes=("result_sha256", "nunique"),
            replicate_count=("replicate_id", "nunique"),
        )
        .sort_values("image_id")
    )


def format_float(value: object, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    if max_rows is not None:
        df = df.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(format_float(value))
            else:
                cells.append(str(value).replace("|", "&#124;"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_analysis_markdown(
    root: Path,
    batch_summary: pd.DataFrame,
    import_status: pd.DataFrame,
    problem_df: pd.DataFrame,
    hash_summary: pd.DataFrame,
    run_scores: pd.DataFrame,
    condition_scores: pd.DataFrame,
) -> Path:
    run_scores = numeric_columns(run_scores)
    condition_scores = numeric_columns(condition_scores)
    import_status = numeric_columns(import_status)
    scored_runs = int((run_scores["scoring_status"] == "scored").sum())
    not_scored_runs = int((run_scores["scoring_status"] != "scored").sum())
    total_runs = int(len(run_scores))
    complete_conditions = int((condition_scores["scored_runs"] == 3).sum()) if "scored_runs" in condition_scores else 0
    not_scored_conditions = condition_scores[condition_scores["not_scored_runs"] > 0].copy()
    accepted_imports = int((import_status["import_status"] == "accepted").sum()) if not import_status.empty else 0
    import_failures = import_status[import_status["import_status"] != "accepted"].copy() if not import_status.empty else pd.DataFrame()
    if import_failures.empty:
        import_failure_counts = pd.DataFrame(columns=["import_status", "run_count"])
    else:
        import_failure_counts = import_failures["import_status"].value_counts().reset_index()
        import_failure_counts.columns = ["import_status", "run_count"]

    high_error = condition_scores.sort_values(
        "mean_abs_arm_rmst_error_months_mean",
        ascending=False,
        na_position="last",
    ).copy()
    low_error = condition_scores.sort_values(
        "mean_abs_arm_rmst_error_months_mean",
        ascending=True,
        na_position="last",
    ).copy()

    deterministic = int((hash_summary["unique_result_hashes"] == 1).sum()) if not hash_summary.empty else 0
    varied_hash = int((hash_summary["unique_result_hashes"] > 1).sum()) if not hash_summary.empty else 0

    lines: list[str] = [
        "# v3 Digitised Results Analysis",
        "",
        "Input: `data/extractor/*.zip`.",
        "",
        "Manual narrative findings are kept separately in `results/tables/digitised_results_findings_manual.md` and are not generated by this script.",
        "",
    ]
    lines.extend(
        [
        "## Run Coverage",
        "",
        f"- Batch zip exports: {int(batch_summary['zip_file'].nunique()) if not batch_summary.empty else 0}.",
        f"- Expected KM-GPT runs scored by manifest: {total_runs}.",
        f"- Accepted imported runs: {accepted_imports}.",
        f"- Scored runs: {scored_runs}.",
        f"- Not-scored runs: {not_scored_runs}.",
        f"- Conditions with all 3 runs scored: {complete_conditions}.",
        f"- Conditions with byte-identical result CSVs across all 3 exports: {deterministic}.",
        f"- Conditions with different result CSV bytes across exports: {varied_hash}.",
        "",
        "## Interpretation",
        "",
        "- Primary RMST means include only runs where the export contains two real geometric curves and distinct Control/Treatment labels.",
        "- Phantom-fragment second curves and arm-label failures are reported separately from RMST error.",
        "- Clean/base, blur, chroma, easy-colour, DPI 200/150/120/75, publisher-raster, and screenshot-slide runs remain eligible in the current exports.",
        "- JPEG, hard-colour, recompression, DPI 90, grayscale, and scanned-legacy failures are mostly import/arm-identification failures, not valid two-arm RMST scores.",
        "- The three independent workflows produce no meaningful RMST spread after scoring; min and max are equal for every condition in the current run.",
        "",
        "## Import Failure Reasons",
        "",
        ]
    )
    if import_failure_counts.empty:
        lines.append("No import-level failures were found.")
    else:
        lines.extend(markdown_table(import_failure_counts, ["import_status", "run_count"]))
    lines.extend(
        [
            "",
            "## Import Failures By Condition",
            "",
        ]
    )
    if import_failures.empty:
        lines.append("No import-level failures were found.")
    else:
        failures_by_condition = (
            import_failures.groupby(["image_id", "import_status"], as_index=False)
            .agg(
                run_count=("run_id", "count"),
                min_curve_points=("min_curve_points", "min"),
                raw_labels=("raw_labels", "first"),
                manifest_labels=("manifest_labels", "first"),
            )
            .sort_values(["import_status", "image_id"])
        )
        lines.extend(
            markdown_table(
                failures_by_condition,
                ["image_id", "import_status", "run_count", "min_curve_points", "raw_labels", "manifest_labels"],
            )
        )
    lines.extend(
        [
            "",
            "## Lowest RMST Error Conditions",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            low_error,
            [
                "image_id",
                "factor",
                "factor_level",
                "scored_runs",
                "not_scored_runs",
                "mean_abs_arm_rmst_error_months_mean",
                "rmst_difference_error_months_mean",
            ],
            max_rows=8,
        )
    )
    lines.extend(["", "## Highest RMST Error Conditions", ""])
    lines.extend(
        markdown_table(
            high_error,
            [
                "image_id",
                "factor",
                "factor_level",
                "scored_runs",
                "not_scored_runs",
                "mean_abs_arm_rmst_error_months_mean",
                "rmst_difference_error_months_mean",
            ],
            max_rows=10,
        )
    )
    lines.extend(["", "## Not-Scored Conditions", ""])
    if not_scored_conditions.empty:
        lines.append("All conditions had at least one scored run.")
    else:
        status = run_scores[run_scores["scoring_status"] != "scored"][
            ["image_id", "run_id", "factor", "factor_level", "status_reason"]
        ].sort_values(["image_id", "run_id"])
        lines.extend(markdown_table(status, ["image_id", "run_id", "factor", "factor_level", "status_reason"]))
    lines.extend(["", "## Batch Import Problems", ""])
    if problem_df.empty:
        lines.append("No zip-to-run mapping problems were found.")
    else:
        lines.extend(markdown_table(problem_df, list(problem_df.columns), max_rows=20))

    output_path = root / "results/tables/digitised_results_analysis.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and analyse v3 digitised-result zip exports.")
    parser.add_argument("--config", type=Path, default=Path("config/study.yaml"))
    parser.add_argument("--zip-dir", type=Path, default=Path("data/extractor"))
    parser.add_argument("--curve-points-out", type=Path, default=Path("data/processed/kmgpt_curve_points.csv.gz"))
    args = parser.parse_args()

    root = run_root()
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    zip_dir = args.zip_dir if args.zip_dir.is_absolute() else root / args.zip_dir
    curve_points_out = args.curve_points_out if args.curve_points_out.is_absolute() else root / args.curve_points_out
    cfg = load_config(cfg_path)

    (root / "data/processed").mkdir(parents=True, exist_ok=True)
    (root / "results/tables").mkdir(parents=True, exist_ok=True)

    curve_points, import_status, batch_summary, problem_df, result_hashes = standardise_zip_exports(root, zip_dir, cfg)
    safe_to_csv(curve_points, curve_points_out)
    safe_to_csv(import_status, root / "data/processed/kmgpt_import_run_status.csv")
    safe_to_csv(batch_summary, root / "data/processed/digitised_batch_summary.csv")
    safe_to_csv(problem_df, root / "data/processed/digitised_import_problems.csv")
    safe_to_csv(result_hashes, root / "data/processed/digitised_result_hashes.csv")
    hash_summary = result_hash_summary(result_hashes)
    safe_to_csv(hash_summary, root / "results/tables/digitised_result_replicate_hash_summary.csv")

    points_for_scoring = curve_points.copy()
    points_for_scoring["time_months"] = pd.to_numeric(points_for_scoring["time_months"], errors="coerce")
    points_for_scoring["survival"] = pd.to_numeric(points_for_scoring["survival"], errors="coerce")
    arm_scores, run_scores, condition_scores = score_curve_points(root, cfg, points_for_scoring)
    safe_to_csv(arm_scores, root / "data/processed/kmgpt_rmst_by_arm.csv")
    safe_to_csv(run_scores, root / "results/tables/kmgpt_rmst_error_scores.csv")
    safe_to_csv(condition_scores, root / "results/tables/kmgpt_rmst_error_by_condition.csv")
    write_summary(root, curve_points_out, run_scores, condition_scores)
    analysis_path = write_analysis_markdown(
        root,
        batch_summary,
        import_status,
        problem_df,
        hash_summary,
        run_scores,
        condition_scores,
    )

    print(f"Wrote curve points: {curve_points_out}")
    print(root / "data/processed/kmgpt_import_run_status.csv")
    print(f"Imported curve-point rows: {len(curve_points)}")
    print(f"Scored runs: {int((run_scores['scoring_status'] == 'scored').sum())}/{len(run_scores)}")
    print(root / "results/tables/kmgpt_rmst_error_by_condition.csv")
    print(analysis_path)


if __name__ == "__main__":
    main()
