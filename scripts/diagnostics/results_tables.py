"""Results broken down by check and by error level, plus the label check and pixel-offset fit.

Outputs (results/diagnostics):
  checks_summary.csv          acceptance outcome counts and which conditions fall in each
  results_by_error_type.csv   per condition: control, treatment, mean absolute arm and TE error
  label_check.csv             per accepted arm: app label vs the arm matched by shape
  accepted_shift_fit.csv      per accepted arm: vertical/horizontal shift that best explains the error
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (TRUTH, FACTOR_ORDER, RAW, RESULTS, manifest, px_per_month, px_per_survival,
                     load_scorer, true_rmst, write)
from km_synthetic_utils import step_survival_at


def checks_summary() -> pd.DataFrame:
    status = pd.read_csv(RAW / "kmgpt_import_run_status.csv")
    rows = []
    for name, group in status.groupby("import_status"):
        conditions = sorted(set(group["image_id"]), key=FACTOR_ORDER.index)
        rows.append({"import_status": name, "runs": len(group), "conditions": "; ".join(conditions)})
    return pd.DataFrame(rows).sort_values("runs", ascending=False)


def results_by_error_type() -> pd.DataFrame:
    by_condition = pd.read_csv(RESULTS / "kmgpt_rmst_error_by_condition.csv").set_index("image_id")
    truth = true_rmst()
    true_effect = truth["treatment"] - truth["control"]
    rows = []
    for image_id in FACTOR_ORDER:
        r = by_condition.loc[image_id]
        scored = int(r["scored_runs"])
        te = r["rmst_difference_error_months_mean"] if scored else np.nan
        rows.append({
            "image_id": image_id,
            "factor": r["factor"],
            "scored_runs": f"{scored}/{int(r['total_runs'])}",
            "control_error": r["control_rmst_error_months_mean"] if scored else np.nan,
            "treatment_error": r["treatment_rmst_error_months_mean"] if scored else np.nan,
            "mean_abs_arm_error": r["mean_abs_arm_rmst_error_months_mean"] if scored else np.nan,
            "te_error": te,
            "te_error_pct_of_true_effect": 100 * te / true_effect if scored else np.nan,
            # Replicates were identical; record the largest spread across them as proof
            "replicate_spread": (r["mean_abs_arm_rmst_error_months_max"] - r["mean_abs_arm_rmst_error_months_min"]) if scored else np.nan,
        })
    out = pd.DataFrame(rows)
    print(f"true RMST control {truth['control']:.3f}, treatment {truth['treatment']:.3f}, effect {true_effect:.3f}")
    return out


def label_check() -> pd.DataFrame:
    by_arm = pd.read_csv(RAW / "kmgpt_rmst_by_arm.csv")
    labels = pd.read_csv(RAW / "kmgpt_curve_points.csv.gz",
                         usecols=["run_id", "extracted_arm_label", "raw_curve_label"]).drop_duplicates()
    merged = by_arm.merge(labels, on=["run_id", "extracted_arm_label"], how="left")
    merged["label_correct"] = merged["raw_curve_label"].str.strip().str.lower() == merged["assigned_arm"].str.lower()
    return merged[["run_id", "image_id", "extracted_arm_label", "raw_curve_label", "assigned_arm",
                   "label_correct", "extracted_rmst_months"]]


def shift_fit() -> pd.DataFrame:
    """Fit each accepted arm as the true curve shifted by dt (months) and dS (survival)."""
    scorer = load_scorer()
    points = pd.read_csv(RAW / "kmgpt_curve_points.csv.gz", usecols=lambda c: c != "notes")
    truth = pd.read_csv(TRUTH / "truth_km_curve.csv")
    by_arm = pd.read_csv(RAW / "kmgpt_rmst_by_arm.csv")
    meta = manifest()
    t = np.arange(0, 36, 0.01) + 0.005
    rows = []
    for image_id in FACTOR_ORDER:
        run = f"{image_id}_rep01"
        arms = by_arm[by_arm["run_id"] == run]
        if arms.empty:
            continue
        dpi = float(meta.loc[image_id, "dpi"])
        for _, a in arms.iterrows():
            g = points[(points["run_id"] == run) & (points["extracted_arm_label"] == a["extracted_arm_label"])]
            ext = step_survival_at(scorer.prepare_step_curve(g), t)
            tr = truth[truth["arm"] == a["assigned_arm"]].sort_values("time_months")
            best = None
            for dt in np.arange(-1.0, 1.0001, 0.01):
                shifted_t = t - dt
                st = np.where(shifted_t < 0, 1.0, step_survival_at(tr, np.clip(shifted_t, 0, None)))
                resid = ext - st
                ds = resid.mean()
                rmse = float(np.sqrt(((resid - ds) ** 2).mean()))
                if best is None or rmse < best[2]:
                    best = (dt, ds, rmse)
            raw = ext - step_survival_at(tr, t)
            periods = [raw[(t >= lo) & (t < lo + 6)].sum() * 0.01 for lo in range(0, 36, 6)]
            rows.append({
                "image_id": image_id, "arm": a["assigned_arm"], "dpi": dpi,
                "rmst_error": a["rmst_error_months"],
                "vertical_shift_survival": best[1], "vertical_shift_px": best[1] * px_per_survival(dpi),
                "horizontal_shift_months": best[0], "horizontal_shift_px": best[0] * px_per_month(dpi),
                "rmse_before_fit": float(np.sqrt((raw ** 2).mean())), "rmse_after_fit": best[2],
                "months_per_px_vertical": TAU_MONTHS / px_per_survival(dpi),
                **{f"error_months_{lo}_{lo + 6}": p for lo, p in zip(range(0, 36, 6), periods)},
            })
    return pd.DataFrame(rows)


TAU_MONTHS = 36.0

if __name__ == "__main__":
    write(checks_summary(), "checks_summary.csv")
    write(results_by_error_type(), "results_by_error_type.csv")
    labels = label_check()
    write(labels, "label_check.csv")
    per_run = labels.groupby("run_id")["label_correct"].all()
    print(f"label check: {int(per_run.sum())} of {len(per_run)} accepted runs have both labels correct")
    write(shift_fit(), "accepted_shift_fit.csv")
