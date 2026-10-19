#  Impact of Image Quality on Automated Kaplan-Meier Curve Digitisation Using Synthetic Known-Truth Survival Curves

This repo reproduces the synthetic two-arm Kaplan–Meier image-quality study used for the current ISPOR poster. It contains the latest primary RMST results and the calculations behind the retained poster figures.

The study has **25 images and 75 observed extraction runs** (three per image). Of these, **39 runs are accepted and 36 rejected**. Failures are part of the dataset. All patient data are simulated. RMST is evaluated at 36 months, with arms matched to truth by curve shape; this can conceal a label swap, which is reported separately in `results/diagnostics/label_check.csv`.

## Windows setup

Tested with 64-bit **Python 3.12.10**. From this folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py analysis
```

Analysis requires only the Python dependencies. To reproduce figures, install **R 4.5.1**, run the R setup below, and install a Windows Poppler build containing `pdftoppm.exe`. Poppler is needed for figures and image generation, not for analysis of the observed exports.

```powershell
Rscript --vanilla scripts/setup_r.R
.\.venv\Scripts\python.exe run.py figures --r-bin 'C:\Program Files\R\R-4.5.1\bin' --poppler-bin 'C:\tools\poppler\Library\bin'
.\.venv\Scripts\python.exe run.py generate --poppler-bin 'C:\tools\poppler\Library\bin'
```

Replace the example installation paths with yours. If R and Poppler are already on `PATH`, omit those options. Arial regular and bold must be available in the Windows Fonts directory; `--font-dir` can select another directory containing those files. See [DEPENDENCIES.md](DEPENDENCIES.md) for tested versions and download links.

## Commands and outputs

| Command | What it does | Default destination |
|---|---|---|
| `python run.py analysis` | Import all three observed batches, classify runs, score RMST, create result/label/shift tables and image-quality metrics | `work/reproduced/` |
| `python run.py figures` | Run analysis, rerun colour clustering, classify pixels, and draw the retained comparison and baseline figures | `work/reproduced/` |
| `python run.py generate` | Simulate trial truth, render and degrade 25 images, and prepare the 75-job upload queue | `work/generated/` |

Every command accepts `--output-dir work/your-run`. Outputs must stay inside this copy's `work/` directory. Generation requires a fresh destination. The bundled `data/` and `results/` files are reference snapshots and are never overwritten by these commands. Rerunning analysis/figures reuses its chosen output destination; close generated CSVs in Excel first.

To verify a completed run:

```powershell
python -B -m unittest discover -s tests -v
python -B tests/verify_reproduction.py --run work/reproduced --generated work/generated --allow-documented-raster-differences
```

Verification fails if requested generated truth or images are missing or differ. The tested Poppler build has two measured raster differences from the original inputs; the flag above allows only those bounded differences. Omit it to require every generated image to match exactly. Consult [VERIFICATION.md](VERIFICATION.md) for the checks and limits.

## Find the evidence

| Location | Contents |
|---|---|
| `config/study.yaml` | Seed, simulated trial, all image degradations, point-count threshold and RMST horizon |
| `data/truth/` | Simulated patient data, KM truth, true RMST, image manifest, run sheet and original generation software metadata |
| `data/images/` | The exact 25 canonical PNGs uploaded for the observed extractions |
| `data/extractor/` | Three curated observed export archives and explicit batch-to-replicate mapping |
| `data/processed/` | Compressed curve points, import statuses, hashes, batch summary and arm scores |
| `results/tables/` | Primary per-run and per-condition RMST results, plus replicate hash counts |
| `results/diagnostics/` | Acceptance, label, shift-fit, image-quality and cluster tables used to explain the poster evidence |
| `results/figures/` | Comparison PNG/SVG, its key and values, full baseline plots with/without a zoom box, and the 4.72 × 5.69 inch original-style baseline PNG/SVG |
| `scripts/core/`, `scripts/diagnostics/`, `scripts/figures/` | Scripts grouped by purpose, with shared utilities in `scripts/` |

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for units and fields, and [OUTPUT_MAP.md](OUTPUT_MAP.md) for each output's inputs and producing script. `MANIFEST.csv` lists the shareable files and SHA-256 checksums; `work/` is excluded.

## Extraction and interpretation

The KM-GPT/LLM extraction application is external and is not bundled. Analyses rerun locally from its **observed exports**, with no API key or paid service required. Image generation does not create new extraction observations. To repeat extraction, use the generated run sheet/upload queue in the external application and export its results separately; new extraction results may differ from this study.

The original batch order is recorded in `data/extractor/batches.csv`, rather than inferred from filenames. Each curated archive retains `batch-summary.csv`, raw `results/` (including metadata), `manifests/`, and `annotated/` overlays. All 75 exported input images were byte-identical to the retained canonical inputs. Duplicate input/prepared images and execution logs are omitted. Original archive hashes identify the source exports; packaged archive hashes identify the curated copies. Raw result CSV bytes are preserved.

Mean absolute arm error is the average of the two absolute RMST errors. Treatment-effect error is `(digitised treatment − control) − (true treatment − control)`, in months. Errors for rejected runs are missing, not zero. Workflow scenarios combine degradations and do not establish single-factor thresholds. The label audit should be read alongside the shape-matched scores.

## Sharing

Project-owned code, synthetic study data and figures are licensed under the [MIT License](LICENSE). This permits reuse, modification and redistribution, including commercial use, provided the copyright and licence notices are retained.

External dependencies retain their own licences; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
