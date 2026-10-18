# Data dictionary

## Shared identifiers and units

- `image_id`: one of the 25 fixed conditions; names encode the baseline or degradation.
- `run_id`: `<image_id>_rep01`, `_rep02`, or `_rep03`.
- `replicate_id`: 1–3, explicitly mapped to an observed archive in `data/extractor/batches.csv`.
- `arm`: `control` or `treatment`. Original extractor labels and curve identifiers are preserved separately.
- Time, medians, RMST and RMST errors are in **months**. `survival` is a **0–1 fraction**. Image dimensions are pixels; DPI is dots per inch; blur sigma is pixels.
- Missing values in error tables indicate an unscored/unavailable estimate. They must not be replaced with zero.

## Inputs

| File(s) | Meaning and important fields |
|---|---|
| `data/truth/base_trial.csv` | 400 simulated patients (200 per arm); patient identifier, arm, event/censoring times, observed time and event indicator. No real patient records. |
| `truth_km_curve.csv` | Known-truth step curves: `arm`, `time_months`, `survival` and risk/event counts. |
| `truth_km_grid.csv` | Truth evaluated on the fixed time grid, used for shape-based arm matching. |
| `truth_summary_by_arm.csv` | One row per arm, including `true_rmst_months` and `rmst_tau_months`. |
| `truth_risk_table.csv` | Simulated at-risk counts. Retained as part of generated truth; the study images have no displayed risk table. |
| `image_manifest.csv` | 25 image definitions. `analysis_block`, `factor`, `factor_level`, `factor_type`, `scenario`, rendering/degradation settings, image dimensions and plot geometry. `image_path` points to a bundled PNG; `pdf_path` identifies a vector file produced only when generating images. |
| `km_gpt_run_sheet.csv` | 75 extraction jobs. In this review copy, repeated jobs point to their canonical image, avoiding 75 duplicate PNGs. Existing completion fields are the original generation checklist, not the observed acceptance record. Generation creates a separate upload queue. |
| `run_metadata.json` | Software metadata recorded for the original synthetic generation. Current reruns record their own environment under `work/`. |
| `data/extractor/batches.csv` | Archive name, replicate, run count, source and curated archive SHA-256, and number of canonical inputs verified. |

Inside each curated archive, `batch-summary.csv` maps extractor-internal IDs to image filenames. `results/<id>.csv` contains traced points (`time`, `St`, `curve`, and identifiers where supplied); `.meta.json` contains curve metadata. `manifests/<id>.json` contains observed image/axis/label readings and confidence metadata. `annotated/<id>.png` is the observed trace overlay.

## Processed observations

| File | Meaning |
|---|---|
| `kmgpt_curve_points.csv.gz` | 91,329 observed points; lossless gzip of the original CSV. Fields: `run_id`, `image_id`, `extracted_arm_label`, `time_months`, `survival`, `notes`, `raw_curve_id`, `raw_curve_label`, `manifest_annotation_label`, `manifest_annotation_source`, `import_status`. Notes retain source archive/result and manifest context. |
| `kmgpt_import_run_status.csv` | One row per observed run, with grouping basis, raw/manifest labels, curve count, minimum usable points, acceptance status and failure reason. Acceptance requires exactly two distinct labelled arms and at least 150 usable points in each. |
| `kmgpt_rmst_by_arm.csv` | 78 accepted arm estimates. `assigned_arm` is matched by curve shape, `extracted_arm_label` preserves the group key, and `extracted_rmst_months`/`rmst_error_months` give the estimate/error. |
| `digitised_batch_summary.csv` | Extractor-supplied status/confidence metadata for all 75 observations. |
| `digitised_result_hashes.csv` | SHA-256 of every original result CSV. Used to check replicate identity without altering observations. |
| `digitised_import_problems.csv` | Empty in this snapshot: every result maps to an expected run. |

## Results

`kmgpt_rmst_error_scores.csv` has one row per run, including `scoring_status`, `status_reason`, arm RMST/errors, `mean_abs_arm_rmst_error_months`, and `rmst_difference_error_months`. The condition table aggregates across replicates with mean/min/max columns, counts scored/rejected runs, and carries factor metadata. Its summary denominator is the scored runs, not all three runs where extraction failed.

Diagnostic tables retain explicit units in their column names. `label_check.csv` compares extractor labels to shape-matched arms. `accepted_shift_fit.csv` fits horizontal/vertical curve offsets; these are explanatory fits. Image-quality tables measure pixel differences against a clean render at the same DPI, along with geometry, JPEG quantisation and palette values. Cluster tables describe per-pixel colour groups and their composition; group numbers are arbitrary and clustering is sampled.

`baseline_comparison_values.csv` records the four illustrated conditions, crop/zoom geometry, pixel-group statistics and relevant observed errors. Figures use replicate 1 traces; the primary tables include all replicates.

`baseline_comparison_svg_validation.json` records vector dimensions and element counts. `baseline_original_4p72x5p69in.json` records the portrait baseline's physical size, pixel dimensions, fonts and displayed layers. These are display settings, separate from the canonical input images used for extraction.
