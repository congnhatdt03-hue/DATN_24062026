"""Step 6: angle matching by MSE."""

import numpy as np


def compute_mse_curve(signature_current, signature_template):
    """Compute circular MSE for all angular shifts."""
    current = np.asarray(signature_current, dtype=np.float32)
    template = np.asarray(signature_template, dtype=np.float32)
    curve = np.zeros(len(template), dtype=np.float32)
    for idx in range(len(template)):
        curve[idx] = float(np.mean((np.roll(current, idx) - template) ** 2))
    return curve


def refine_angle_parabolic(mse_curve, best_index):
    """Refine best angle using parabolic interpolation around the minimum."""
    n = len(mse_curve)
    left = float(mse_curve[(best_index - 1) % n])
    mid = float(mse_curve[best_index])
    right = float(mse_curve[(best_index + 1) % n])
    denom = left - 2.0 * mid + right
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denom
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return 0.0
    return float(offset)


def match_by_mse(signature_current, signature_template, invert_angle=False):
    """Match two signatures and return best angle.

    `invert_angle` dao chieu goc tra ve de chot quy uoc dau sau khi kiem chung
    bang mot stator xoay goc da biet (vd anh xoay vat ly 90 do). Goc duoc tinh
    ket hop noi suy parabol quanh cuc tieu nen co do phan giai duoi 1 do.
    """
    curve = compute_mse_curve(signature_current, signature_template)
    best_index = int(np.argmin(curve))
    offset = refine_angle_parabolic(curve, best_index)
    angle_step = 360.0 / float(len(curve))
    coarse_angle_deg = best_index * angle_step
    refined_angle_deg = (best_index + offset) * angle_step
    if invert_angle:
        coarse_angle_deg = -coarse_angle_deg
        refined_angle_deg = -refined_angle_deg
    coarse_angle_deg = float(coarse_angle_deg % 360.0)
    refined_angle_deg = float(refined_angle_deg % 360.0)
    min_error = float(curve[best_index])
    return {
        "success": True,
        "angle_deg": refined_angle_deg,
        "coarse_angle_deg": coarse_angle_deg,
        "refined_angle_deg": refined_angle_deg,
        "best_index": best_index,
        "parabolic_offset_step": float(offset),
        "parabolic_offset_deg": float(offset * angle_step),
        "min_error": min_error,
        "mse_curve": curve,
        "logs": [
            "Best MSE index: {} (+offset {:.3f})".format(best_index, offset),
            "Goc tho: {:.3f} deg | Goc tinh: {:.3f} deg (invert={})".format(
                coarse_angle_deg,
                refined_angle_deg,
                invert_angle,
            ),
            "Min error: {:.6f}".format(min_error),
        ],
    }
