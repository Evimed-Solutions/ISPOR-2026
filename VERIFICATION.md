# Reproduction checks

Checked on Windows on 18 October 2026 using the versions recorded in `DEPENDENCIES.md`.

| Check | Result |
|---|---|
| Observed data | All 25 conditions, 75 runs and 91,329 curve points retained |
| Acceptance | 39 accepted; 24 rejected for shared labels, 6 for missing labels, 6 for fewer than 150 points per arm |
| Numerical reproduction | All 24 retained CSV/gzip tables match, including schemas, row order, identifiers and statuses; numeric tolerance is absolute `1e-9`, with no relative tolerance |
| Headline errors | Baseline 0.050 months; scored mean-arm-error range 0.030–0.291; heaviest-blur treatment-effect error 0.451 months |
| Poster figure reproduction | All five retained PNGs are pixel-identical; the native-vector comparison SVG is byte-identical |
| Visual inspection | Regenerated comparison and portrait baseline checked for crop, callouts, axes, traces, original-style legend and physical dimensions; existing full-plot geometry matches the canonical baseline pixel for pixel |
| Synthetic generation | All five truth CSVs reproduce exactly; 25 images and 75 upload jobs generated |
| Export curation | All 75 exported input images were byte-identical to the retained canonical images; raw result CSV bytes and hashes are preserved |
| Standalone rerun | Analysis rerun from an exported copy with Python access to the parent research workspace blocked; all 19 analysis-mode tables match |
| Package checks | All 11 tests passed, covering evidence integrity, acceptance boundaries, protected outputs, fresh figure startup and verification failures |
| Fresh figure run | The complete figure command succeeded directly into a new output directory, including R preflight, clustering and both added SVG exporters |

## Reproduction limits

The original Poppler version was not recorded. With tested Poppler 26.09.0, **23 of 25 generated images are pixel-identical** to the observed inputs. `dpi_200.png` differs in **1 of 1,228,800 pixels** (maximum RGB-channel difference 24); `workflow_publisher_raster_export.png` differs in **45 pixels** (maximum difference 22). The bundled original images are preserved and remain the inputs for scoring, diagnostics and poster reproduction. Newly generated images are kept separate and do not replace observed data.

The original colour-clustering random seed was not recorded. This copy sets diagnostic seed 260619; in the tested environment it reproduces every retained cluster table and figure exactly. Repeating extraction through the external application is a new experiment and is not covered by these checks.

The reference SVGs are included and regenerated. The comparison SVG is deterministic and checked byte for byte. Matplotlib SVG metadata and element identifiers can vary; the matching PNGs check the displayed content of those plots. The portrait baseline is exactly 4.72 × 5.69 inches (1416 × 1707 pixels at 300 DPI), with 19 pt tick/legend fonts and 21 pt axis labels.

## Repeat the checks

Run `python -B -m unittest discover -s tests -v` for archive/result integrity, observation completeness, label-audit preservation, acceptance boundaries, protected output paths, manifest checksums, fresh temporary-directory setup and generation-failure handling.

After `python run.py figures` and `python run.py generate`, run:

```powershell
python -B tests/verify_reproduction.py --run work/reproduced --generated work/generated --allow-documented-raster-differences
```

This writes `work/reproduced/verification.json` with per-table and per-image comparisons. Mismatched primary results, figures, requested generated truth, missing outputs and unexpected generated images return a failing exit code. The optional flag permits only `dpi_200.png` to differ by at most 1 pixel with a maximum channel change of 24, and `workflow_publisher_raster_export.png` by at most 45 pixels with a maximum channel change of 22. Dimensions must match. Omit the flag for strict pixel identity. Differences are reported even when this explicit allowance is used. `MANIFEST.csv` verifies the bundled reference inputs, scripts and documentation.
