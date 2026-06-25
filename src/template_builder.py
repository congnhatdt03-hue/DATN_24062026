"""Step 5: template generation."""

import copy
import json
from pathlib import Path

from .io_utils import read_image, write_image
from .tab_edge_filter import filter_tab_edges
from .radial_signature import build_radial_signature, draw_radial_rays, plot_signature_image


def _pick_edge_image(images, source_mode):
    """Pick the preferred debug edge image without ndarray truthiness issues."""
    for key in (source_mode, "closed_edges", "tab_edges_clean"):
        if key in images and images[key] is not None:
            return images[key]
    return None


def build_template_from_roi(roi, center, radius, tab_edge_params, radial_params):
    """Build template data from one ROI."""
    tab_result = filter_tab_edges(roi, center, radius, tab_edge_params)
    if not tab_result["success"]:
        return {"success": False, "data": {}, "images": tab_result.get("images", {}), "logs": tab_result.get("logs", [])}
    source_mode = str(radial_params.get("source_mode", "closed_edges"))
    source_image = _pick_edge_image(tab_result["images"], source_mode)
    radial_result = build_radial_signature(source_image, center, radial_params, radius=radius)
    if not radial_result["success"]:
        return {
            "success": False,
            "data": {},
            "images": {
                **tab_result["images"],
                "radial_source": radial_result["data"]["radial_source"],
                "radius_band": radial_result["data"]["radius_band"],
                "radial_rays": draw_radial_rays(
                    roi,
                    center,
                    radial_result["data"]["signature_raw"],
                    radius=radius,
                    measured_mask=radial_result["data"]["measured_mask"],
                ),
                "signature_plot": plot_signature_image(radial_result["data"]["signature_raw"], radial_result["data"]["signature_norm"]),
            },
            "logs": tab_result["logs"] + radial_result["logs"],
        }
    template_data = {
        "signature_raw": radial_result["data"]["signature_raw"].tolist(),
        "signature_norm": radial_result["data"]["signature_norm"].tolist(),
        "center": [float(center[0]), float(center[1])],
        "radius": float(radius),
        "tab_edge_params": tab_edge_params,
        "radial_params": radial_params,
        "note": "0 degree template",
    }
    images = {
        "radial_rays": draw_radial_rays(
            roi,
            center,
            radial_result["data"]["signature_raw"],
            radius=radius,
            measured_mask=radial_result["data"]["measured_mask"],
        ),
        "radial_source": radial_result["data"]["radial_source"],
        "radius_band": radial_result["data"]["radius_band"],
        **tab_result["images"],
        "signature_plot": plot_signature_image(radial_result["data"]["signature_raw"], radial_result["data"]["signature_norm"]),
        "template_roi": roi.copy(),
    }
    return {"success": True, "data": template_data, "images": images, "logs": tab_result["logs"] + radial_result["logs"]}


def save_template_data(path, template_data):
    """Save template JSON."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(template_data, handle, ensure_ascii=False, indent=2)
    return path_obj


def load_template_data(path):
    """Load template JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_template_roi_path(template_data_path, template_data):
    """Resolve the saved ROI image path for one template JSON file."""
    roi_path = template_data.get("template_roi_path")
    if not roi_path:
        return None
    roi_path_obj = Path(roi_path)
    if not roi_path_obj.is_absolute():
        roi_path_obj = Path(template_data_path).resolve().parent / roi_path_obj
    return roi_path_obj


def save_template_bundle(template_data_path, template_data, template_roi_image=None, template_roi_path=None, store_relative_roi_path=False):
    """Save template JSON and, if available, the ROI image that belongs to it."""
    template_data_path = Path(template_data_path)
    payload = copy.deepcopy(template_data)
    roi_target = None

    if template_roi_image is not None:
        roi_target = Path(template_roi_path) if template_roi_path is not None else template_data_path.with_name("{}_roi.png".format(template_data_path.stem))
        write_image(roi_target, template_roi_image)
        if store_relative_roi_path and roi_target.parent == template_data_path.parent:
            payload["template_roi_path"] = roi_target.name
        else:
            payload["template_roi_path"] = str(roi_target)

    save_template_data(template_data_path, payload)
    return payload, template_data_path, roi_target


def load_template_bundle(template_data_path):
    """Load template JSON plus its saved ROI image if the image exists."""
    template_path = Path(template_data_path)
    template_data = load_template_data(template_path)
    roi_path = resolve_template_roi_path(template_path, template_data)
    template_roi_image = read_image(roi_path) if roi_path else None
    if roi_path is not None:
        template_data["template_roi_path"] = str(roi_path)
    return template_data, template_roi_image, roi_path


def run_template_step(roi, center, radius, tab_edge_params, radial_params):
    """Run template creation step."""
    return build_template_from_roi(roi, center, radius, tab_edge_params, radial_params)
