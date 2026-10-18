"""Labels every clustered pixel by what it really is and measures how cleanly the arms separate.

Each pixel from dump_clusters.R is looked up in the clean image at the same DPI (same
geometry; for the palette conditions the baseline gives the arm positions) and labelled as
control curve, treatment curve, legend swatch, legend text, other dark marks, a pixel that is
white in the clean image ("new"), or a blend.

Outputs:
  results/diagnostics/cluster_composition.csv        per image and pixel type: share in each cluster
  results/diagnostics/cluster_purity_by_condition.csv per image: arm pixels kept, share in own cluster, red hue wrap
  results/diagnostics/hsl_by_condition.csv            per image: median hue, mean saturation and lightness per arm
  intermediates/classified/<image_id>.csv          labelled pixels, input to experimental analyses (excluded from this package)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

from _common import TRUTH, OUT, CLEAN_REFERENCE, FACTOR_ORDER, IMAGES, RAW, WORK, manifest, write

ORDER = ["Control", "Treatment", "Legend swatch C", "Legend swatch T", "Legend text",
         "Dark text/ticks", "New (white in clean)", "Blend"]
# Legend box as a fraction of the image (upper right of the plot area)
LEGEND_X, LEGEND_Y = (0.735, 0.93), (0.12, 0.215)


def classify(ref_px: np.ndarray, rows: np.ndarray, cols: np.ndarray, height: int, width: int) -> np.ndarray:
    r, g, b = ref_px[:, 0], ref_px[:, 1], ref_px[:, 2]
    mx, mn = ref_px.max(axis=1), ref_px.min(axis=1)
    cls = np.full(len(ref_px), "Blend", dtype=object)
    cls[b > r + 40] = "Control"                          # blue in the clean image
    cls[(r > g + 40) & (r > b + 40)] = "Treatment"       # red in the clean image
    cls[(mx - mn < 30) & (mx < 200)] = "Dark text/ticks"
    cls[(mn >= 200) & (mx - mn < 30)] = "New (white in clean)"
    legend = ((cols > LEGEND_X[0] * width) & (cols < LEGEND_X[1] * width)
              & (rows > LEGEND_Y[0] * height) & (rows < LEGEND_Y[1] * height))
    out = cls.copy()
    out[legend & (cls == "Control")] = "Legend swatch C"
    out[legend & (cls == "Treatment")] = "Legend swatch T"
    out[legend & (cls == "Dark text/ticks")] = "Legend text"
    return out


def cluster_names(summary_row: pd.Series) -> dict[int, str]:
    names = {}
    for k in (1, 2):
        if summary_row[f"cluster{k}_saturation"] == 0:
            names[k] = f"grey L{summary_row[f'cluster{k}_lightness']:.2f}"
        else:
            names[k] = f"hue {summary_row[f'cluster{k}_hue']:.0f}"
    return names


def main() -> None:
    meta = manifest()
    summary = pd.read_csv(OUT / "cluster_summary.csv").set_index("image_id")
    status = pd.read_csv(RAW / "kmgpt_import_run_status.csv")
    status = status[status["replicate_id"] == 1].set_index("image_id")["import_status"]
    out_dir = WORK / "classified"
    out_dir.mkdir(parents=True, exist_ok=True)

    composition, purity, hsl = [], [], []
    for image_id in FACTOR_ORDER:
        ref = np.asarray(Image.open(IMAGES / f"{CLEAN_REFERENCE[int(meta.loc[image_id, 'dpi'])]}.png").convert("RGB")).astype(int)
        height, width = ref.shape[:2]
        d = pd.read_csv(WORK / "pixels" / f"{image_id}_pixels.csv")
        rows = (height - d["arr_row"]).clip(0, height - 1).to_numpy()
        cols = (d["arr_col"] - 1).to_numpy()
        d["cls"] = classify(ref[rows, cols], rows, cols, height, width)
        d.to_csv(out_dir / f"{image_id}.csv", index=False)

        names = cluster_names(summary.loc[image_id])
        table = pd.crosstab(d["cls"], d["group"])
        for cls in ORDER:
            if cls in table.index:
                n = int(table.loc[cls].sum())
                row = {"image_id": image_id, "pixel_type": cls, "pixels": n}
                for k in table.columns:
                    row[f"share_in_cluster{k}_pct"] = 100 * table.loc[cls, k] / n
                    row[f"cluster{k}"] = names[k]
                composition.append(row)

        ctrl, trt = d[d["cls"] == "Control"], d[d["cls"] == "Treatment"]
        c_share = ctrl["group"].value_counts(normalize=True)
        t_share = trt["group"].value_counts(normalize=True)
        same_majority = (len(trt) > 0 and len(ctrl) > 0 and c_share.idxmax() == t_share.idxmax())
        # Where do wrapped red pixels sit? Distance to the nearest control (blue) curve pixel.
        wrapped = trt[trt["h"] > 180]
        near_blue = np.nan
        if len(wrapped) and len(ctrl):
            dist, _ = cKDTree(ctrl[["arr_row", "arr_col"]].to_numpy()).query(wrapped[["arr_row", "arr_col"]].to_numpy())
            near_blue = 100 * float(np.mean(dist <= 3))
        purity.append({
            "image_id": image_id,
            "control_pixels": len(ctrl), "treatment_pixels": len(trt),
            "control_in_own_cluster_pct": 100 * c_share.max() if len(ctrl) else np.nan,
            "treatment_in_own_cluster_pct": 100 * t_share.max() if len(trt) else np.nan,
            "red_hue_wrapped_pct": 100 * (trt["h"] > 180).mean() if len(trt) else np.nan,
            "red_hue_wrapped_pixels": len(wrapped),
            "wrapped_within_3px_of_blue_pct": near_blue,
            "new_pixels": int((d["cls"] == "New (white in clean)").sum()),
            "arms_majority": "merged" if same_majority else "separate",
            "outcome": status[image_id],
        })
        hsl.append({
            "image_id": image_id,
            "control_hue_median": ctrl["h"].median(), "control_saturation_mean": ctrl["s"].mean(),
            "control_lightness_mean": ctrl["l"].mean(),
            "treatment_hue_median": trt["h"].median() if len(trt) else np.nan,
            "treatment_saturation_mean": trt["s"].mean() if len(trt) else np.nan,
            "treatment_lightness_mean": trt["l"].mean() if len(trt) else np.nan,
            "legend_text_lightness_mean": d.loc[d["cls"] == "Legend text", "l"].mean(),
        })
        print(f"{image_id:34s} {names[1]:>10s} / {names[2]:<10s} red wrap {purity[-1]['red_hue_wrapped_pct']:5.1f}%")

    write(pd.DataFrame(composition), "cluster_composition.csv")
    write(pd.DataFrame(purity), "cluster_purity_by_condition.csv")
    write(pd.DataFrame(hsl), "hsl_by_condition.csv")


if __name__ == "__main__":
    main()
