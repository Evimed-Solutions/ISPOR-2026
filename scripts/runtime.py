"""Resolve package resources and keep generated files inside the package's work folder."""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def run_root() -> Path:
    root = Path(os.environ.get("ISPOR_RUN_ROOT", PACKAGE_ROOT / "work/reproduced")).resolve()
    work = (PACKAGE_ROOT / "work").resolve()
    if root == work or not root.is_relative_to(work):
        raise ValueError("Output must be a subdirectory of this package's work folder")
    return root
