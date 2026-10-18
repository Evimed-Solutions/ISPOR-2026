# Tested environment

- Windows 11, Python **3.12.10**, R **4.5.1**.
- Python direct dependencies are pinned to the tested versions in `requirements.txt`. The package uses no web requests or model API calls during reproduction.
- R `SurvdigitizeR` **0.0.0.9000**, installed from `Pechli-Lab/SurvdigitizeR` commit **b9d2ad9500311189ba8adc2ec57831ad8b6042d5**.
- R `cluster` **2.1.8.1**. `scripts/setup_r.R` installs the direct R dependencies; it needs internet access and may require Rtools for source packages.
- Poppler Windows **26.09.0**, portable build **v26.09.0-0**, was used to verify this copy. The original generation metadata recorded Poppler availability but not its version.
- Arial regular and bold from Windows. Fonts are read from the Windows installation and are not redistributed.

Sources: [SurvdigitizeR repository](https://github.com/Pechli-Lab/SurvdigitizeR), [cluster package](https://cran.r-project.org/package=cluster), [Poppler Windows releases](https://github.com/oschwartz10612/poppler-windows/releases/tag/v26.09.0-0).

For the tested Poppler ZIP (`Release-26.09.0-0.zip`), SHA-256 is `7a6f256a0ddf7536182246a5733331bf4677cbcc34f4663774947ad34556c8d0`. Download and extract it outside the committed package, then point `--poppler-bin` at its `Library/bin` directory. No external software binaries are bundled in the shareable files.

Changing rasteriser, fonts or image libraries can change a few pixels. The observed canonical images and observed extractor exports remain the primary analysis inputs. R colour clustering uses sampled medoids; this copy sets seed 260619 for repeatable new diagnostic runs. The original diagnostic run did not record its random seed. See `VERIFICATION.md` for the measured comparisons.
