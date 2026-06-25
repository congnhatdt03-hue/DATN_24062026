"""Step 1: HoughCircle detection."""

import copy
import math

import cv2
import numpy as np

from .preprocess import preprocess_for_hough
from .visualization import draw_circles


def _scale_hough_params(params, scale):
    """Scale length-based Hough params when processing on a resized image."""
    scaled = copy.deepcopy(params)
    hough_cfg = scaled.get("hough", {})
    for key in ("minDist", "minRadius", "maxRadius", "min_center_dist"):
        if key in hough_cfg:
            hough_cfg[key] = max(1.0, float(hough_cfg[key]) * scale)
    filter_cfg = scaled.get("filter", {})
    if "radius_consensus_tol" in filter_cfg:
        filter_cfg["radius_consensus_tol"] = max(1, int(round(float(filter_cfg["radius_consensus_tol"]) * scale)))
    return scaled


def _prepare_fast_mode_input(image, params):
    """Optionally resize the tray image for faster Hough detection."""
    fast_cfg = params.get("fast_mode", {})
    if not fast_cfg.get("enabled", False):
        return image, params, 1.0, []

    height, width = image.shape[:2]
    try:
        max_dim = int(round(float(fast_cfg.get("max_processing_dim", 1400))))
    except (TypeError, ValueError):
        max_dim = 1400
    max_dim = max(320, max_dim)
    if max(height, width) <= max_dim:
        return image, params, 1.0, []

    scale = float(max_dim) / float(max(height, width))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    proc_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    logs = [
        "Fast mode: detect tren anh thu nho {}x{} ({:.0f}% anh goc).".format(new_size[0], new_size[1], scale * 100.0),
        "Scale tam va ban kinh ve anh goc truoc khi cat ROI.",
    ]
    return proc_image, _scale_hough_params(params, scale), scale, logs


def _scale_circle_to_original(circle, scale_back):
    """Scale one detected circle from resized-image space back to original space."""
    return {
        **circle,
        "x": int(round(float(circle["x"]) * scale_back)),
        "y": int(round(float(circle["y"]) * scale_back)),
        "r": max(1, int(round(float(circle["r"]) * scale_back))),
    }


def _edge_score(edges, cx, cy, radius, ring_width=3):
    radius = int(round(radius))
    if radius < 6:
        return 0.0
    offsets = []
    for ring_radius in range(max(1, radius - ring_width), radius + ring_width + 1):
        sample_count = max(96, int(round(2.0 * math.pi * ring_radius)))
        angles = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
        xs = np.rint(ring_radius * np.cos(angles)).astype(np.int32)
        ys = np.rint(ring_radius * np.sin(angles)).astype(np.int32)
        offsets.append(np.stack((xs, ys), axis=1))
    offsets = np.unique(np.concatenate(offsets, axis=0), axis=0)
    sample_x = cx + offsets[:, 0]
    sample_y = cy + offsets[:, 1]
    valid = (
        (sample_x >= 0)
        & (sample_x < edges.shape[1])
        & (sample_y >= 0)
        & (sample_y < edges.shape[0])
    )
    valid_count = int(valid.sum())
    if valid_count <= 0:
        return 0.0
    return float((edges[sample_y[valid], sample_x[valid]] > 0).sum()) / float(valid_count)


def detect_hough_candidates(image, params):
    """Detect raw Hough candidates."""
    proc_image, proc_params, processing_scale, fast_logs = _prepare_fast_mode_input(image, params)
    preprocess_result = preprocess_for_hough(proc_image, proc_params)
    gray = preprocess_result["data"]
    edges = cv2.Canny(gray, 70, 170)
    hough_cfg = proc_params.get("hough", {})
    base_param2 = int(round(float(hough_cfg.get("param2", 38))))
    p2_values = [max(8, base_param2 - 10), max(8, base_param2 - 5), base_param2, base_param2 + 5]
    candidates = []
    for param2 in p2_values:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=max(1.0, float(hough_cfg.get("dp", 1.2))),
            minDist=max(10.0, float(hough_cfg.get("minDist", 120))),
            param1=max(1.0, float(hough_cfg.get("param1", 110))),
            param2=max(1.0, float(param2)),
            minRadius=max(1, int(round(float(hough_cfg.get("minRadius", 32))))),
            maxRadius=max(2, int(round(float(hough_cfg.get("maxRadius", 140))))),
        )
        if circles is None:
            continue
        for cx, cy, radius in np.round(circles[0]).astype(int):
            score = _edge_score(edges, cx, cy, radius)
            candidates.append({"x": int(cx), "y": int(cy), "r": int(radius), "score": float(score)})
    if processing_scale < 1.0:
        scale_back = 1.0 / float(processing_scale)
        candidates = [_scale_circle_to_original(circle, scale_back) for circle in candidates]
    logs = list(fast_logs) + list(preprocess_result["logs"])
    logs.append("So ung vien Hough truoc loc: {}".format(len(candidates)))
    images = {"preprocessed": gray, "edges": edges}
    return {
        "success": True,
        "data": candidates,
        "images": images,
        "logs": logs,
        "meta": {
            "processing_scale": processing_scale,
            "processing_size": (proc_image.shape[1], proc_image.shape[0]),
        },
    }


def filter_duplicate_circles(circles, params):
    """Remove circles whose centers are too close."""
    min_center_dist = float(params.get("hough", {}).get("min_center_dist", 90))
    ordered = sorted(circles, key=lambda item: item.get("score", 0.0), reverse=True)
    kept = []
    for circle in ordered:
        if any(math.hypot(circle["x"] - other["x"], circle["y"] - other["y"]) < min_center_dist for other in kept):
            continue
        kept.append(circle)
    return kept


def filter_circles_by_radius_consensus(circles, params):
    """Keep circles near the dominant radius."""
    if not circles:
        return [], None
    filter_cfg = params.get("filter", {})
    if not filter_cfg.get("use_radius_consensus", True):
        return list(circles), float(np.median([item["r"] for item in circles]))
    tol = max(1, int(round(float(filter_cfg.get("radius_consensus_tol", 6)))))
    bins = {}
    for circle in circles:
        key = int(round(float(circle["r"]) / float(tol)))
        bins.setdefault(key, []).append(circle)
    dominant = max(bins.values(), key=lambda group: (len(group), sum(item["score"] for item in group)))
    common_radius = float(np.median([item["r"] for item in dominant]))
    filtered = [item for item in circles if abs(item["r"] - common_radius) <= tol]
    if filter_cfg.get("force_common_radius", True):
        filtered = [dict(item, r=int(round(common_radius))) for item in filtered]
    return filtered, common_radius


def select_expected_by_grid(circles, expected, radius, min_score=0.0):
    """Keep `expected` circles best matching the tray grid layout.

    Truoc tien loai cac ung vien co diem canh `score` < `min_score`: day la
    cac vong tron rac tren nen/luoi/san (ring it diem canh that). Nho do khi
    mot stator that bi thieu (vi du bat rong phan quang Hough khong thay),
    buoc chon khong "lap day" bang vong rac ma tra ve so luong dung thuc te.

    Sau khi loc, neu so circle van vuot `expected`, uu tien giu cac tam co
    nhieu lang gieng thang hang: cung hang (gan nhau theo y) va cung cot (gan
    nhau theo x). Stator that nam tren luoi deu nen co ca row va col support
    cao; vong rac le thuong chi thang hang mot chieu hoac don le. Diem canh
    `score` dung de pha hoa khi grid support bang nhau.
    """
    candidates = [item for item in circles if float(item.get("score", 0.0)) >= float(min_score)]
    if expected <= 0 or len(candidates) <= expected:
        return candidates
    tol = max(1.0, float(radius) * 0.55)
    ranked = []
    for circle in candidates:
        row_support = sum(
            1 for other in circles if other is not circle and abs(other["y"] - circle["y"]) < tol
        )
        col_support = sum(
            1 for other in circles if other is not circle and abs(other["x"] - circle["x"]) < tol
        )
        grid_key = (min(row_support, col_support), row_support + col_support, float(circle.get("score", 0.0)))
        ranked.append((grid_key, circle))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:expected]]


def sort_circles_by_grid(circles):
    """Sort circles by rows then columns and assign IDs."""
    if not circles:
        return []
    sorted_circles = sorted(circles, key=lambda item: (item["y"], item["x"]))
    row_gap = max(40.0, float(np.median([item["r"] for item in sorted_circles])) * 1.4)
    rows = []
    for circle in sorted_circles:
        if not rows:
            rows.append([circle])
            continue
        row_mean = float(np.mean([item["y"] for item in rows[-1]]))
        if abs(circle["y"] - row_mean) <= row_gap:
            rows[-1].append(circle)
        else:
            rows.append([circle])
    ordered = []
    next_id = 1
    for row in rows:
        for circle in sorted(row, key=lambda item: item["x"]):
            ordered.append(dict(circle, id=next_id))
            next_id += 1
    return ordered


def run_hough_step(image, params):
    """Run the full Hough step."""
    candidate_result = detect_hough_candidates(image, params)
    circles_all = candidate_result["data"]
    deduped = filter_duplicate_circles(circles_all, params)
    consensus, common_radius = filter_circles_by_radius_consensus(deduped, params)
    expected = int(round(float(params.get("expected_count", 12))))
    min_score = float(params.get("filter", {}).get("min_score", 0.1))
    grid_radius = common_radius if common_radius else (consensus[0]["r"] if consensus else 0)
    selected = select_expected_by_grid(consensus, expected, grid_radius, min_score)
    circles_filtered = sort_circles_by_grid(selected)
    logs = list(candidate_result["logs"])
    logs.append("So ung vien sau loc trung tam: {}".format(len(deduped)))
    logs.append("So circle sau loc ban kinh: {}".format(len(consensus)))
    if len(consensus) > len(selected):
        logs.append("Loai {} vong rac (score < {:.2f} hoac ngoai luoi).".format(len(consensus) - len(selected), min_score))
    logs.append("So circle cuoi cung: {}".format(len(circles_filtered)))
    if len(circles_filtered) != expected:
        logs.append("Canh bao: phat hien {} / {} stator".format(len(circles_filtered), expected))
    if common_radius is not None:
        logs.append("Ban kinh dong thuan ~ {:.1f}px".format(common_radius))
    images = {
        "original": image.copy(),
        "preprocessed": candidate_result["images"]["preprocessed"],
        "hough_all": draw_circles(image, sort_circles_by_grid(deduped), color=(0, 165, 255), adaptive_style=True),
        "hough_filtered": draw_circles(image, circles_filtered, color=(0, 255, 0), adaptive_style=True),
    }
    return {
        "success": True,
        "data": {
            "circles_all": sort_circles_by_grid(deduped),
            "circles_filtered": circles_filtered,
            "processing_scale": float(candidate_result.get("meta", {}).get("processing_scale", 1.0)),
            "processing_size": candidate_result.get("meta", {}).get("processing_size"),
        },
        "images": images,
        "logs": logs,
    }
