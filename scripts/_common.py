"""Shared helpers for standalone step-test scripts."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_HOUGH_PARAMS,
    DEFAULT_ROI_PARAMS,
    HOUGH_PRESET_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    ROI_DIR,
    ROI_PRESET_PATH,
    TEMPLATE_DATA_PATH,
    TEMPLATE_DIR,
)
from src.io_utils import ensure_project_dirs, get_first_image, read_image  # noqa: E402
from src.pipeline_runner import (  # noqa: E402
    run_full_pipeline,
    run_step_hough,
    run_step_matching,
    run_step_radial,
    run_step_roi,
    run_step_tab_edges,
    run_step_template,
)
from src.preset_store import ensure_default_presets, load_preset, load_radial_signature_preset  # noqa: E402
from src.roi_extractor import save_rois  # noqa: E402
from src.template_builder import load_template_data, save_template_data  # noqa: E402
from src.visualization import save_debug_images  # noqa: E402


def bootstrap():
    """Ensure standard folders and presets exist."""
    ensure_project_dirs()
    ensure_default_presets()


def load_standard_presets():
    """Load all standard presets with defaults merged."""
    radial_signature = load_radial_signature_preset()
    return {
        "hough": load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS),
        "roi": load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS),
        "tab_edge": radial_signature["tab_edge_params"],
        "radial": radial_signature["radial_params"],
    }


def print_logs(result):
    """Print step logs in a readable way."""
    for line in result.get("logs", []):
        print("-", line)


def save_result_images(step_name, result):
    """Save returned debug images to data/output/<step_name>/."""
    images = result.get("images", {})
    if not images:
        return None
    output_dir = OUTPUT_DIR / step_name
    saved = save_debug_images(output_dir, images)
    if saved:
        print("Da luu debug images vao:", output_dir)
    return saved


def get_first_tray_image():
    """Return the first tray image in data/input or None."""
    return get_first_image(INPUT_DIR)


def _manual_roi_item(image_path, roi_id=1):
    """Build a simple ROI item from a standalone ROI image."""
    image = read_image(image_path)
    if image is None:
        return None
    height, width = image.shape[:2]
    radius = min(width, height) * 0.35
    return {
        "id": int(roi_id),
        "roi": image,
        "offset_x": 0,
        "offset_y": 0,
        "center_in_roi": (width / 2.0, height / 2.0),
        "radius": radius,
        "circle": {"id": int(roi_id), "x": width / 2.0, "y": height / 2.0, "r": radius},
    }


def derive_first_roi_from_tray():
    """Run Hough + ROI on the first tray image and return the first ROI item."""
    tray_path = get_first_tray_image()
    if tray_path is None:
        return None, [
            "Chua co anh trong data/input/.",
            "Hay dat anh khay vao data/input/ roi chay lai script nay.",
        ]

    presets = load_standard_presets()
    hough_result = run_step_hough(str(tray_path), presets["hough"])
    if not hough_result["success"]:
        return None, hough_result.get("logs", []) or ["Buoc Hough that bai."]

    roi_result = run_step_roi(
        hough_result["images"]["original"],
        hough_result["data"]["circles_filtered"],
        presets["roi"],
    )
    if not roi_result["success"] or not roi_result["data"].get("rois"):
        return None, roi_result.get("logs", []) or ["Buoc ROI that bai."]

    return roi_result["data"]["rois"][0], []


def get_first_roi_item(prefer_template=False):
    """Get an ROI item from data/template, data/roi, or derive one from tray image."""
    if prefer_template:
        template_image = get_first_image(TEMPLATE_DIR)
        if template_image is not None:
            roi_item = _manual_roi_item(template_image, roi_id=1)
            if roi_item is not None:
                return roi_item, []

    roi_image = get_first_image(ROI_DIR)
    if roi_image is not None:
        roi_item = _manual_roi_item(roi_image, roi_id=1)
        if roi_item is not None:
            return roi_item, []

    return derive_first_roi_from_tray()


def get_template_data_or_help():
    """Load template_data.json or return a guidance message list."""
    if not TEMPLATE_DATA_PATH.is_file():
        return None, [
            "Chua co presets/template_data.json.",
            "Hay chay: python scripts/run_template_step.py",
            "hoac tao template trong GUI truoc.",
        ]
    try:
        return load_template_data(TEMPLATE_DATA_PATH), []
    except Exception as exc:
        return None, ["Khong doc duoc template_data.json: {}".format(exc)]


def save_template_if_success(result):
    """Save template_data.json if the template step succeeded."""
    if result.get("success") and result.get("data"):
        save_template_data(TEMPLATE_DATA_PATH, result["data"])
        print("Da luu template tai:", TEMPLATE_DATA_PATH)


def print_missing(messages):
    """Print a clear guidance block and return gracefully."""
    for line in messages:
        print(line)


def save_rois_if_success(result):
    """Save extracted ROIs if the ROI step succeeded."""
    if result.get("success") and result.get("data", {}).get("rois"):
        saved = save_rois(result["data"]["rois"], ROI_DIR)
        print("Da luu {} ROI vao: {}".format(len(saved), ROI_DIR))
        return saved
    return []
