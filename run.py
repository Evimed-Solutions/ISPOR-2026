"""Reproduce the bundled study without changing its reference inputs or results."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from runtime import run_root

PYTHON_PACKAGES = ("numpy", "pandas", "scipy", "matplotlib", "Pillow", "PyYAML", "scikit-image")


def check_dependencies(mode: str, env: dict) -> dict:
    versions, missing = {}, []
    for package in PYTHON_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError("Missing Python dependencies: " + ", ".join(missing) + ". Install requirements.txt")
    if mode in ("figures", "generate"):
        poppler = shutil.which("pdftoppm", path=env["PATH"])
        if not poppler:
            raise RuntimeError("Poppler pdftoppm.exe is missing. Add its bin folder to PATH or use --poppler-bin.")
        result = subprocess.run([poppler, "-v"], capture_output=True, text=True, check=True)
        versions["Poppler"] = (result.stderr or result.stdout).splitlines()[0]
    if mode == "figures":
        rscript = shutil.which("Rscript", path=env["PATH"])
        if not rscript:
            raise RuntimeError("Rscript.exe is missing. Add R's bin folder to PATH or use --r-bin.")
        fonts = Path(env.get("ISPOR_FONT_DIR", str(Path(env.get("WINDIR", "C:/Windows")) / "Fonts")))
        for name in ("arial.ttf", "arialbd.ttf"):
            if not (fonts / name).is_file():
                raise RuntimeError(f"Required Windows font {name} is missing. Use --font-dir.")
        code = ('for(p in c("SurvdigitizeR","cluster")) { '
                'if(!requireNamespace(p,quietly=TRUE)) stop(paste("Missing R package",p)); '
                'cat(p,as.character(packageVersion(p)),"\\n") }')
        result = subprocess.run([rscript, "--vanilla", "-e", code], env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr + result.stdout)
        versions["R_packages"] = result.stdout.strip().splitlines()
    return versions


def prepare_inputs(destination: Path) -> None:
    """Stage only package inputs. No paths refer to the parent research workspace."""
    for relative in ("data/truth", "data/images", "data/extractor", "config"):
        shutil.copytree(ROOT / relative, destination / relative, dirs_exist_ok=True)
    for relative in ("data/processed", "results/tables", "results/diagnostics", "results/figures", "intermediates"):
        (destination / relative).mkdir(parents=True, exist_ok=True)


def execute(relative: str, destination: Path, env: dict) -> None:
    program = ROOT / "scripts" / relative
    if program.suffix == ".R":
        command = [shutil.which("Rscript", path=env["PATH"]), "--vanilla", str(program)]
    else:
        command = [sys.executable, "-B", str(program)]
    print(f"\nRunning {relative}", flush=True)
    subprocess.run(command, cwd=destination, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analysis", "figures", "generate"))
    parser.add_argument("--output-dir", help="Subdirectory inside work; defaults to work/reproduced or work/generated")
    parser.add_argument("--poppler-bin", type=Path, help="Directory containing pdftoppm.exe")
    parser.add_argument("--r-bin", type=Path, help="Directory containing Rscript.exe")
    parser.add_argument("--font-dir", type=Path, help="Directory containing Windows Arial fonts")
    args = parser.parse_args()
    destination = Path(args.output_dir or ("work/generated" if args.mode == "generate" else "work/reproduced"))
    if not destination.is_absolute():
        destination = ROOT / destination
    os.environ["ISPOR_RUN_ROOT"] = str(destination.resolve())
    destination = run_root()
    env = os.environ.copy()
    extra_bins = [str(p.resolve()) for p in (args.poppler_bin, args.r_bin) if p]
    env["PATH"] = os.pathsep.join(extra_bins + [env.get("PATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(str(ROOT / "scripts" / p) for p in ("", "core", "diagnostics", "figures"))
    env["MPLCONFIGDIR"] = str(destination / "intermediates/matplotlib")
    env["TEMP"] = env["TMP"] = str(destination / "intermediates/temp")
    if args.font_dir:
        env["ISPOR_FONT_DIR"] = str(args.font_dir.resolve())
    # R needs a usable temporary directory even for its dependency check.
    (destination / "intermediates/temp").mkdir(parents=True, exist_ok=True)
    versions = check_dependencies(args.mode, env)
    if args.mode == "generate" and (destination / "data/images").exists():
        raise FileExistsError("Generation needs a fresh output directory; use --output-dir work/generated-new")
    if args.mode == "generate":
        (destination / "config").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "config/study.yaml", destination / "config/study.yaml")
    else:
        prepare_inputs(destination)
    if args.mode == "generate":
        execute("core/generate_images.py", destination, env)
    else:
        for stage in ("core/import_results.py", "diagnostics/results_tables.py", "diagnostics/image_quality.py"):
            execute(stage, destination, env)
        if args.mode == "figures":
            for stage in ("diagnostics/dump_clusters.R", "diagnostics/classify_clusters.py",
                          "figures/baseline_comparison.py", "figures/full_baseline.py",
                          "figures/comparison_svg.py", "figures/baseline_portrait.py"):
                execute(stage, destination, env)
    metadata = {"mode": args.mode, "python": sys.version.split()[0], "dependencies": versions,
                "output": destination.relative_to(ROOT).as_posix()}
    (destination / "run_environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nCompleted: {destination.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
