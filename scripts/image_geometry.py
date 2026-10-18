"""Detect plot axes and translate time/survival coordinates into pixel crops."""
import math
import numpy as np
from _common import AXIS_WIDTH, AXIS_HEIGHT, AXIS_LEFT, AXIS_BOTTOM, TAU, Y_TOP

def spine_centres(profile: np.ndarray, threshold: float) -> list[float]:
    idx = np.where(profile > threshold)[0]
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [float(g.mean()) for g in groups]


def find_frame(arr: np.ndarray) -> dict[str, float]:
    """Plot frame from the axis spines (long dark lines); returns pixel centres of the four spines."""
    h, w = arr.shape[:2]
    lum = arr[..., :3] @ np.array([0.299, 0.587, 0.114])
    dark = lum < 110
    rows = spine_centres(dark.sum(axis=1), 0.5 * AXIS_WIDTH * w)
    cols = spine_centres(dark.sum(axis=0), 0.5 * AXIS_HEIGHT * h)
    frame = {"top": rows[0], "bottom": rows[-1], "left": cols[0], "right": cols[-1]}
    expected = {"left": AXIS_LEFT * w, "right": (AXIS_LEFT + AXIS_WIDTH) * w,
                "top": (1 - AXIS_BOTTOM - AXIS_HEIGHT) * h, "bottom": (1 - AXIS_BOTTOM) * h}
    for k in frame:
        if abs(frame[k] - expected[k]) > 4:
            raise SystemExit(f"frame {k} {frame[k]:.1f} px differs from expected {expected[k]:.1f} px")
    return frame


def to_px(frame, t=None, s=None):
    if t is not None:
        return frame["left"] + t / TAU * (frame["right"] - frame["left"])
    return frame["bottom"] - s / Y_TOP * (frame["bottom"] - frame["top"])


def crop_box(frame, t_range, s_range):
    x0, x1 = math.floor(to_px(frame, t=t_range[0])), math.ceil(to_px(frame, t=t_range[1]))
    y0, y1 = math.floor(to_px(frame, s=s_range[1])), math.ceil(to_px(frame, s=s_range[0]))
    return x0, y0, x1, y1
