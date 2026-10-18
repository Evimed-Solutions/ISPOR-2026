# Output map

Run commands from the repository root. Generated counterparts are written below the command's output directory, preserving the same data/results organisation.

| Output(s) | Producing script | Required inputs |
|---|---|---|
| Synthetic trial, KM truth, image manifest, 25 PNGs, vector PDFs and 75-job upload queue | `scripts/core/generate_images.py` via `run.py generate` | `config/study.yaml`, NumPy seed, Poppler |
| Curve points, import status, batch summary, result hashes, import problems | `scripts/core/import_results.py` | Curated ZIP exports, explicit batch map, run sheet and settings |
| Arm RMST and per-run/per-condition RMST tables | Importer using `scripts/core/score_rmst.py` | Imported points/status, known truth, manifest, run sheet |
| Replicate result-hash summary | Importer | Raw result hashes |
| `checks_summary.csv`, `results_by_error_type.csv`, `label_check.csv`, `accepted_shift_fit.csv` | `scripts/diagnostics/results_tables.py` | Primary tables, points/status, truth, geometry |
| `image_quality_metrics.csv`, `dpi_geometry.csv`, `blur_kernel.csv`, `jpeg_quantisation.csv`, `recompression_steps.csv`, `palette_colours.csv` | `scripts/diagnostics/image_quality.py` | Canonical PNGs, manifest and primary condition scores |
| `cluster_summary.csv` and intermediate pixel dumps | `scripts/diagnostics/dump_clusters.R` | Canonical PNGs, SurvdigitizeR and cluster |
| `cluster_composition.csv`, `cluster_purity_by_condition.csv`, `hsl_by_condition.csv` | `scripts/diagnostics/classify_clusters.py` | Cluster dumps, clean images, manifest/status |
| `baseline_comparison.png`, `baseline_comparison_legend.png`, `baseline_comparison_values.csv` | `scripts/figures/baseline_comparison.py` | Replicate 1 points, images, cluster dumps, error and quality tables |
| `full_plot_baseline.png/.svg`, `full_plot_baseline_zoombox.png/.svg`, `full_plot_baseline_validation.json` | `scripts/figures/full_baseline.py` | Original trial/KM truth, baseline points, comparison zoom values, Poppler |
| `baseline_comparison.svg`, `baseline_comparison_svg_validation.json` | `scripts/figures/comparison_svg.py` | Replicate 1 points, images, reproduced cluster dumps and image-quality table |
| `baseline_original_4p72x5p69in.png/.svg/.json` | `scripts/figures/baseline_portrait.py` | Trial/KM truth and study settings; original censor marks and upper-right legend |

Shared numerical routines are in `scripts/core/km_synthetic_utils.py`. Paths and plot constants are in `scripts/_common.py` and `scripts/runtime.py`; the retained figure helpers are in `scripts/figure_helpers.py` and `scripts/image_geometry.py`. These helpers replace dependencies on earlier, excluded figure drafts.

The runner stages inputs, checks dependencies and executes stages in order. Its analysis output also includes generated technical summaries in `results/tables/`; they are not copies of working write-ups. `run_environment.json` records the runtime used. Large intermediates and generated upload copies stay under ignored `work/`.
