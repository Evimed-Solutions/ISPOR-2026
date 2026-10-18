from __future__ import annotations


import hashlib
import math

import numpy as np
import pandas as pd


ARM_CONTROL = "control"
ARM_TREATMENT = "treatment"
ARMS = (ARM_CONTROL, ARM_TREATMENT)


def weibull_scale_from_median(median: float, shape: float) -> float:
    return median / (math.log(2.0) ** (1.0 / shape))


def trial_truth_params(cfg: dict) -> dict[str, dict[str, float]]:
    shape = float(cfg["trial"]["weibull_shape"])
    medians = cfg["trial"]["median_months"]
    return {
        arm: {
            "shape": shape,
            "scale": weibull_scale_from_median(float(medians[arm]), shape),
            "median": float(medians[arm]),
        }
        for arm in ARMS
    }


def simulate_trial(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    trial_cfg = cfg["trial"]
    n = int(trial_cfg["n_per_arm"])
    params = trial_truth_params(cfg)
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        shape = params[arm]["shape"]
        scale = params[arm]["scale"]
        event_time = scale * rng.weibull(shape, size=n)
        dropout = rng.uniform(
            float(trial_cfg["dropout_min_months"]),
            float(trial_cfg["dropout_max_months"]),
            size=n,
        )
        censor_time = np.minimum(dropout, float(trial_cfg["admin_followup_months"]))
        observed_time = np.minimum(event_time, censor_time)
        event = event_time <= censor_time
        for idx in range(n):
            rows.append(
                {
                    "patient_id": f"{arm}_{idx + 1:03d}",
                    "arm": arm,
                    "observed_time_months": float(observed_time[idx]),
                    "event": int(event[idx]),
                    "true_event_time_months": float(event_time[idx]),
                    "censor_time_months": float(censor_time[idx]),
                }
            )
    return pd.DataFrame(rows)


def km_step_curve(df_arm: pd.DataFrame, horizon_months: float) -> pd.DataFrame:
    durations = df_arm["observed_time_months"].to_numpy(float)
    events = df_arm["event"].to_numpy(int)
    event_times = np.sort(np.unique(durations[(events == 1) & (durations <= horizon_months)]))
    times = [0.0]
    survival = [1.0]
    current = 1.0
    for t in event_times:
        at_risk = np.sum(durations >= t)
        d = np.sum((durations == t) & (events == 1))
        if at_risk > 0:
            current *= 1.0 - d / at_risk
            times.append(float(t))
            survival.append(float(current))
    if times[-1] < horizon_months:
        times.append(float(horizon_months))
        survival.append(float(current))
    return pd.DataFrame({"time_months": times, "survival": survival})


def step_survival_at(step_df: pd.DataFrame, query_times: np.ndarray) -> np.ndarray:
    times = step_df["time_months"].to_numpy(float)
    survival = step_df["survival"].to_numpy(float)
    idx = np.searchsorted(times, query_times, side="right") - 1
    idx = np.clip(idx, 0, len(survival) - 1)
    return survival[idx]


def build_km_outputs(trial_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon = float(cfg["trial"]["admin_followup_months"])
    grid_step = float(cfg["extraction"]["grid_step_months"])
    grid = np.arange(0.0, horizon + 1e-9, grid_step)
    curve_rows: list[pd.DataFrame] = []
    grid_rows: list[pd.DataFrame] = []
    risk_rows: list[dict[str, object]] = []
    risk_times = np.array(cfg["trial"]["risk_table_times_months"], dtype=float)
    for arm in ARMS:
        df_arm = trial_df[trial_df["arm"] == arm]
        step_df = km_step_curve(df_arm, horizon)
        step_df = step_df.assign(arm=arm)
        curve_rows.append(step_df)
        grid_rows.append(
            pd.DataFrame(
                {
                    "arm": arm,
                    "time_months": grid,
                    "survival": step_survival_at(step_df, grid),
                }
            )
        )
        durations = df_arm["observed_time_months"].to_numpy(float)
        for t in risk_times:
            risk_rows.append({"arm": arm, "time_months": t, "at_risk": int(np.sum(durations >= t))})
    return pd.concat(curve_rows, ignore_index=True), pd.concat(grid_rows, ignore_index=True), pd.DataFrame(risk_rows)


def km_step_median(step_df: pd.DataFrame) -> float:
    crossed = step_df[step_df["survival"] <= 0.5]
    if crossed.empty:
        return float("nan")
    return float(crossed.iloc[0]["time_months"])






def rmst_from_step(step_df: pd.DataFrame, tau: float) -> float:
    curve = step_df[["time_months", "survival"]].dropna().sort_values("time_months")
    if curve.empty:
        return float("nan")
    times = curve["time_months"].to_numpy(float)
    survival = curve["survival"].to_numpy(float)
    if times[0] > 0:
        times = np.insert(times, 0, 0.0)
        survival = np.insert(survival, 0, survival[0])
    if not np.isclose(times, tau).any():
        s_tau = float(step_survival_at(pd.DataFrame({"time_months": times, "survival": survival}), np.array([tau]))[0])
        insert_at = int(np.searchsorted(times, tau, side="right"))
        times = np.insert(times, insert_at, tau)
        survival = np.insert(survival, insert_at, s_tau)
    keep = times <= tau
    times = times[keep]
    survival = survival[keep]
    if len(times) < 2:
        return 0.0
    return float(np.sum(np.diff(times) * survival[:-1]))


def stable_rng(cfg: dict, *parts: object) -> np.random.Generator:
    key = "|".join([str(cfg["seed"]), *[str(part) for part in parts]])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    return np.random.default_rng(seed)
