"""Visualization helpers for OpenCV images."""

import csv
from pathlib import Path

import cv2
import numpy as np

from .io_utils import ensure_dir, write_image


def normalize_display_angle_deg(angle_deg):
    """Convert an angle to the user-facing signed range [-180, 180]."""
    angle = ((float(angle_deg) + 180.0) % 360.0) - 180.0
    if abs(angle + 180.0) < 1e-9 and float(angle_deg) > 0.0:
        return 180.0
    return float(angle)


def cv_bgr_to_rgb(image):
    """Convert BGR or grayscale OpenCV image to RGB."""
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_for_display(image, max_width=1200, max_height=900):
    """Resize an image to fit a display area."""
    if image is None:
        return None
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    scale = min(float(max_width) / float(w), float(max_height) / float(h), 1.0)
    if scale >= 1.0:
        return image.copy()
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _circle_style(image, adaptive_style=False):
    """Return drawing sizes for circle overlays."""
    if image is None:
        return {
            "circle_thickness": 2,
            "marker_size": 14,
            "marker_thickness": 2,
            "center_radius": 4,
            "font_scale": 0.55,
            "score_font_scale": 0.45,
            "text_thickness": 2,
            "score_thickness": 1,
        }
    if not adaptive_style:
        return {
            "circle_thickness": 2,
            "marker_size": 14,
            "marker_thickness": 2,
            "center_radius": 4,
            "font_scale": 0.55,
            "score_font_scale": 0.45,
            "text_thickness": 2,
            "score_thickness": 1,
        }
    max_dim = max(image.shape[:2])
    return {
        "circle_thickness": max(2, int(round(max_dim / 400.0))),
        "marker_size": max(14, int(round(max_dim / 55.0))),
        "marker_thickness": max(2, int(round(max_dim / 500.0))),
        "center_radius": max(4, int(round(max_dim / 220.0))),
        "font_scale": max(0.55, float(max_dim) / 1800.0),
        "score_font_scale": max(0.45, float(max_dim) / 2200.0),
        "text_thickness": max(2, int(round(max_dim / 750.0))),
        "score_thickness": max(1, int(round(max_dim / 1100.0))),
    }


def _draw_text_with_outline(image, text, org, font_scale, color, thickness):
    """Draw readable text with a dark outline."""
    outline = max(1, thickness + 2)
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        outline,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_center_axes_overlay(image, center=None, margin_ratio=0.05):
    """Overlay a clean XY axis view centered on the image (or a provided point)."""
    if image is None:
        return None
    if len(image.shape) == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    height, width = base.shape[:2]
    if height <= 0 or width <= 0:
        return base

    if center is None:
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
    cx = int(round(float(center[0])))
    cy = int(round(float(center[1])))
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))

    margin_x = max(12, int(round(width * float(margin_ratio))))
    margin_y = max(12, int(round(height * float(margin_ratio))))
    x_left = (margin_x, cy)
    x_right = (max(margin_x, width - margin_x - 1), cy)
    y_top = (cx, margin_y)
    y_bottom = (cx, max(margin_y, height - margin_y - 1))

    max_dim = max(height, width)
    thickness = max(2, int(round(max_dim / 380.0)))
    outline = thickness + 2
    font_scale = max(0.48, float(max_dim) / 1550.0)
    text_thickness = max(1, int(round(thickness * 0.8)))
    tip_length = 0.024

    x_color = (0, 186, 255)
    y_color = (255, 214, 64)
    neg_x_color = (90, 140, 170)
    neg_y_color = (120, 150, 95)
    origin_color = (255, 255, 255)

    overlay = base.copy()

    cv2.line(overlay, x_left, (cx, cy), (18, 18, 18), outline, cv2.LINE_AA)
    cv2.line(overlay, (cx, cy), x_right, (18, 18, 18), outline, cv2.LINE_AA)
    cv2.line(overlay, (cx, cy), y_top, (18, 18, 18), outline, cv2.LINE_AA)
    cv2.line(overlay, (cx, cy), y_bottom, (18, 18, 18), outline, cv2.LINE_AA)

    cv2.line(overlay, x_left, (cx, cy), neg_x_color, thickness, cv2.LINE_AA)
    cv2.arrowedLine(overlay, (cx, cy), x_right, x_color, thickness, cv2.LINE_AA, tipLength=tip_length)
    cv2.arrowedLine(overlay, (cx, cy), y_top, y_color, thickness, cv2.LINE_AA, tipLength=tip_length)
    cv2.line(overlay, (cx, cy), y_bottom, neg_y_color, thickness, cv2.LINE_AA)

    cv2.circle(overlay, (cx, cy), outline + 1, (18, 18, 18), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), max(3, thickness + 1), origin_color, -1, cv2.LINE_AA)

    x_label_pos = (max(4, x_right[0] - 34), max(20, cy - 10))
    y_label_pos = (min(width - 26, cx + 10), max(22, y_top[1] + 20))
    origin_label_pos = (min(width - 28, cx + 8), min(height - 8, cy + 20))
    _draw_text_with_outline(overlay, "+X", x_label_pos, font_scale, x_color, text_thickness)
    _draw_text_with_outline(overlay, "+Y", y_label_pos, font_scale, y_color, text_thickness)
    _draw_text_with_outline(overlay, "O", origin_label_pos, font_scale * 0.92, origin_color, text_thickness)

    return cv2.addWeighted(overlay, 0.92, base, 0.08, 0)


def draw_circles(image, circles, color=(0, 255, 0), draw_ids=True, adaptive_style=False):
    """Draw detected circles on an image."""
    output = image.copy()
    style = _circle_style(output, adaptive_style=adaptive_style)
    for index, circle in enumerate(circles, start=1):
        x = int(round(circle.get("x", circle.get("center_x", 0))))
        y = int(round(circle.get("y", circle.get("center_y", 0))))
        r = int(round(circle.get("r", circle.get("radius", 0))))
        score = float(circle.get("score", 0.0))
        cv2.circle(output, (x, y), r, color, style["circle_thickness"], cv2.LINE_AA)
        cv2.circle(output, (x, y), style["center_radius"] + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, (x, y), style["center_radius"], (0, 255, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(
            output,
            (x, y),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            style["marker_size"],
            style["marker_thickness"],
        )
        if draw_ids:
            label = "ID{:02d}".format(circle.get("id", index))
            label_x = max(8, x - int(round(style["marker_size"] * 1.5)))
            label_y = max(24, y - r - int(round(style["marker_size"] * 0.7)))
            _draw_text_with_outline(
                output,
                label,
                (label_x, label_y),
                style["font_scale"],
                color,
                style["text_thickness"],
            )
            score_y = min(output.shape[0] - 10, y + r + int(round(style["marker_size"] * 0.8)))
            _draw_text_with_outline(
                output,
                "s={:.2f}".format(score),
                (label_x, score_y),
                style["score_font_scale"],
                (0, 255, 255),
                style["score_thickness"],
            )
    return output


def draw_roi_boxes(image, roi_items):
    """Draw ROI boxes on an image."""
    output = image.copy()
    for item in roi_items:
        x1 = int(item["offset_x"])
        y1 = int(item["offset_y"])
        crop_width = int(round(float(item.get("crop_width", item["roi"].shape[1]))))
        crop_height = int(round(float(item.get("crop_height", item["roi"].shape[0]))))
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            output,
            "ID{:02d}".format(item["id"]),
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
        )
    return output


def draw_final_results(image, results):
    """Draw final circle, ID, and angle results on the source image.

    Chu va ky hieu tu co gian theo kich thuoc anh khay de van doc duoc ID khi anh
    lon bi thu nho hien thi. ID ve to, can giua phia tren moi stator.
    """
    output = image.copy()
    style = _circle_style(output, adaptive_style=True)
    id_font_scale = style["font_scale"] * 1.3
    for result in results:
        x = int(round(result["center_x"]))
        y = int(round(result["center_y"]))
        r = int(round(result["radius"]))
        angle_deg = float(result.get("angle_deg", 0.0))
        angle_display_deg = normalize_display_angle_deg(angle_deg)
        status = str(result.get("status", "ok"))
        ring_color = (0, 255, 0) if status == "ok" else (0, 0, 255)
        cv2.circle(output, (x, y), r, ring_color, style["circle_thickness"], cv2.LINE_AA)
        cv2.circle(output, (x, y), style["center_radius"] + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, (x, y), style["center_radius"], (0, 255, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(output, (x, y), (0, 0, 255), cv2.MARKER_CROSS, style["marker_size"], style["marker_thickness"])
        tip_x = int(round(x + r * np.cos(np.deg2rad(angle_deg))))
        tip_y = int(round(y + r * np.sin(np.deg2rad(angle_deg))))
        cv2.arrowedLine(output, (x, y), (tip_x, tip_y), (0, 165, 255), max(2, style["circle_thickness"]), cv2.LINE_AA, tipLength=0.2)

        id_label = "ID{:02d}".format(int(result["id"]))
        (text_w, text_h), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, style["text_thickness"])
        id_x = max(4, x - text_w // 2)
        id_y = max(text_h + 6, y - r - int(round(style["marker_size"] * 0.5)))
        _draw_text_with_outline(output, id_label, (id_x, id_y), id_font_scale, (0, 255, 255), style["text_thickness"])

        angle_label = "{:.1f} deg {}".format(angle_display_deg, status)
        angle_y = min(output.shape[0] - 8, y + r + int(round(style["marker_size"] * 0.9)))
        _draw_text_with_outline(output, angle_label, (max(4, x - r), angle_y), style["font_scale"], (255, 255, 0), style["text_thickness"])
    return output


def make_debug_grid(images, labels=None, cell_size=(420, 320)):
    """Create a simple image grid from named debug images."""
    valid_items = [(name, image) for name, image in images.items() if image is not None]
    if not valid_items:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    labels = labels or {}
    prepared = []
    for name, image in valid_items:
        display = image
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        display = cv2.resize(display, cell_size, interpolation=cv2.INTER_AREA)
        cv2.putText(
            display,
            labels.get(name, name),
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        prepared.append(display)

    cols = 2
    rows = int(np.ceil(len(prepared) / float(cols)))
    blank = np.zeros_like(prepared[0])
    row_images = []
    for row_idx in range(rows):
        row_slice = prepared[row_idx * cols : (row_idx + 1) * cols]
        while len(row_slice) < cols:
            row_slice.append(blank.copy())
        row_images.append(np.hstack(row_slice))
    return np.vstack(row_images)


def make_pair_view(left_image, right_image, left_label="", right_label="", cell_height=360):
    """Ghep 2 anh canh nhau de so sanh truc quan (vd stator mau vs stator test)."""

    def _prepare(image, label):
        if image is None:
            panel = np.zeros((cell_height, cell_height, 3), dtype=np.uint8)
        else:
            display = image
            if len(display.shape) == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            height, width = display.shape[:2]
            scale = float(cell_height) / float(max(1, height))
            new_width = max(1, int(round(width * scale)))
            panel = cv2.resize(display, (new_width, cell_height), interpolation=cv2.INTER_AREA)
        if label:
            _draw_text_with_outline(panel, label, (12, 28), 0.7, (0, 255, 255), 2)
        return panel

    left = _prepare(left_image, left_label)
    right = _prepare(right_image, right_label)
    separator = np.full((cell_height, 4, 3), 60, dtype=np.uint8)
    return np.hstack([left, separator, right])


def save_debug_images(output_dir, images):
    """Save a dictionary of debug images to a folder."""
    out_dir = ensure_dir(output_dir)
    saved = {}
    for name, image in images.items():
        if image is None:
            continue
        path = out_dir / "{}.png".format(name)
        write_image(path, image)
        saved[name] = str(path)
    return saved


def save_results_csv(path, results):
    """Save final pipeline rows to CSV."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    fieldnames = ["id", "center_x", "center_y", "radius", "angle_deg", "min_error", "status"]
    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path_obj
