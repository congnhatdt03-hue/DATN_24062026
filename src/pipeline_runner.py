"""Orchestrate individual steps and the full pipeline."""

import time
from pathlib import Path

from .angle_matcher import match_by_mse
from .config import OUTPUT_DIR
from .hough_detector import run_hough_step
from .io_utils import read_image, write_image
from .radial_signature import build_radial_signature, draw_radial_rays, plot_signature_image
from .roi_extractor import run_roi_crop_step, run_roi_refine_step, run_roi_step
from .tab_edge_filter import filter_tab_edges
from .template_builder import run_template_step
from .visualization import draw_final_results, save_debug_images, save_results_csv


def _pick_edge_image(tab_images, source_mode):
    """Pick the preferred edge image without relying on ndarray truthiness."""
    for key in (source_mode, "closed_edges", "tab_edges_clean"):
        if key in tab_images and tab_images[key] is not None:
            return tab_images[key]
    return None


def _read_image_input(image_path_or_image):
    if isinstance(image_path_or_image, (str, Path)):
        image = read_image(str(image_path_or_image))
        if image is None:
            raise ValueError("Khong doc duoc anh: {}".format(image_path_or_image))
        return image
    return image_path_or_image


def run_step_hough(image_path_or_image, hough_params):
    """Run step 1 from image path or image."""
    try:
        image = _read_image_input(image_path_or_image)
        return run_hough_step(image, hough_params)
    except Exception as exc:
        return {"success": False, "data": {}, "images": {}, "logs": [str(exc)]}


def run_step_roi(image, circles, roi_params, selected_id=None, refine_mode="selected", save_all=True):
    """Run step 2."""
    try:
        return run_roi_step(
            image,
            circles,
            roi_params,
            selected_id=selected_id,
            refine_mode=refine_mode,
            save_all=save_all,
        )
    except Exception as exc:
        return {"success": False, "data": {}, "images": {}, "logs": [str(exc)]}


def run_step_roi_crop(image, circles, roi_params, save_all=True):
    """Crop all ROI items from the tray image."""
    try:
        return run_roi_crop_step(image, circles, roi_params, save_all=save_all)
    except Exception as exc:
        return {"success": False, "data": {}, "images": {}, "logs": [str(exc)]}


def run_step_roi_refine(roi_item, roi_params):
    """Refine one already-cropped ROI item."""
    try:
        return run_roi_refine_step(roi_item, roi_params)
    except Exception as exc:
        return {"success": False, "data": {}, "images": {}, "logs": [str(exc)]}


def run_step_tab_edges(roi_item, tab_edge_params):
    """Run step 3."""
    try:
        return filter_tab_edges(roi_item["roi"], roi_item["center_in_roi"], roi_item["radius"], tab_edge_params)
    except Exception as exc:
        return {"success": False, "data": {}, "images": {}, "logs": [str(exc)]}


def run_step_radial(roi_item, tab_edges_clean, radial_params):
    """Run step 4."""
    try:
        if isinstance(tab_edges_clean, dict):
            tab_images = tab_edges_clean
        else:
            tab_images = {"tab_edges_clean": tab_edges_clean}
        source_mode = str(radial_params.get("source_mode", "closed_edges"))
        source_image = _pick_edge_image(tab_images, source_mode)
        radial_result = build_radial_signature(
            source_image,
            roi_item["center_in_roi"],
            radial_params,
            radius=roi_item["radius"],
        )
        radial_result["images"] = {
            "radial_rays": draw_radial_rays(
                roi_item["roi"],
                roi_item["center_in_roi"],
                radial_result["data"]["signature_raw"],
                radius=roi_item["radius"],
                params=radial_params,
                measured_mask=radial_result["data"]["measured_mask"],
            ),
            "radial_source": radial_result["data"]["radial_source"],
            "radius_band": radial_result["data"]["radius_band"],
            **tab_images,
            "signature_plot": plot_signature_image(
                radial_result["data"]["signature_raw"],
                radial_result["data"]["signature_norm"],
            ),
        }
        return radial_result
    except Exception as exc:
        return {"success": False, "data": {}, "images": {}, "logs": [str(exc)]}


def run_step_template(roi_item, tab_edge_params, radial_params):
    """Run step 5."""
    try:
        return run_template_step(
            roi_item["roi"],
            roi_item["center_in_roi"],
            roi_item["radius"],
            tab_edge_params,
            radial_params,
        )
    except Exception as exc:
        return {"success": False, "data": {}, "images": {}, "logs": [str(exc)]}


def run_step_matching(roi_item, template_data, tab_edge_params, radial_params):
    """Run step 6."""
    try:
        tab_result = run_step_tab_edges(roi_item, tab_edge_params)
        if not tab_result["success"]:
            return tab_result
        radial_result = run_step_radial(roi_item, tab_result["images"], radial_params)
        if not radial_result["success"]:
            return {
                "success": False,
                "data": {},
                "images": {**tab_result["images"], **radial_result["images"]},
                "logs": tab_result["logs"] + radial_result["logs"],
            }
        match_result = match_by_mse(
            radial_result["data"]["signature_norm"],
            template_data["signature_norm"],
            invert_angle=bool(radial_params.get("invert_angle", False)),
        )
        return {
            "success": True,
            "data": {
                "angle_deg": match_result["angle_deg"],
                "coarse_angle_deg": match_result["coarse_angle_deg"],
                "refined_angle_deg": match_result["refined_angle_deg"],
                "min_error": match_result["min_error"],
                "mse_curve": match_result["mse_curve"],
                "signature_raw": radial_result["data"]["signature_raw"],
                "signature_norm": radial_result["data"]["signature_norm"],
            },
            "images": {**tab_result["images"], **radial_result["images"]},
            "logs": tab_result["logs"] + radial_result["logs"] + match_result["logs"],
        }
    except Exception as exc:
        return {"success": False, "data": {}, "images": {}, "logs": [str(exc)]}


def _match_roi_for_pipeline(roi_item, template_norm, tab_edge_params, radial_params, invert_angle):
    """Lean tab-edge -> radial -> MSE for one ROI in the full pipeline.

    Bo qua viec ve radial_rays va do thi matplotlib (chi can goc + sai so) de chay
    nhanh tren ca khay 12 stator. Van giu lai `tab_edges_clean` de xem debug theo ID.
    """
    tab_result = filter_tab_edges(roi_item["roi"], roi_item["center_in_roi"], roi_item["radius"], tab_edge_params)
    tab_clean = tab_result.get("images", {}).get("tab_edges_clean")
    if not tab_result["success"]:
        return {"success": False, "tab_edges_clean": tab_clean}
    source_image = _pick_edge_image(tab_result["images"], str(radial_params.get("source_mode", "closed_edges")))
    radial_result = build_radial_signature(
        source_image,
        roi_item["center_in_roi"],
        radial_params,
        radius=roi_item["radius"],
    )
    if not radial_result["success"]:
        return {"success": False, "tab_edges_clean": tab_clean}
    match_result = match_by_mse(
        radial_result["data"]["signature_norm"],
        template_norm,
        invert_angle=invert_angle,
    )
    return {
        "success": True,
        "angle_deg": float(match_result["angle_deg"]),
        "min_error": float(match_result["min_error"]),
        "tab_edges_clean": tab_clean,
    }


def run_full_pipeline(image_path, hough_params, roi_params, tab_edge_params, radial_params, template_data, save_debug=False):
    """Run the whole tray pipeline (steps 1-6) on one tray image.

    Toi uu toc do: coarse Hough (fast mode neu bat o preset) -> cat ROI tu anh goc ->
    Hough refine -> tab edge -> radial -> MSE, dung matcher gon khong ve matplotlib.
    Mac dinh khong ghi anh debug (`save_debug=False`) de uu tien thoi gian; van luon
    xuat `final_result.png` va `results.csv`.
    """
    start = time.perf_counter()
    logs = []
    image = read_image(image_path)
    if image is None:
        return {"success": False, "data": {}, "images": {}, "logs": ["Khong doc duoc anh khay."]}
    if not template_data or "signature_norm" not in template_data:
        return {"success": False, "data": {}, "images": {}, "logs": ["Thieu template (signature_norm). Hay tao o tab 5."]}

    hough_result = run_hough_step(image, hough_params)
    logs.extend(hough_result["logs"])
    if not hough_result["success"]:
        return {"success": False, "data": {}, "images": hough_result["images"], "logs": logs}

    roi_result = run_roi_step(
        image,
        hough_result["data"]["circles_filtered"],
        roi_params,
        refine_mode="all",
        save_all=save_debug,
    )
    logs.extend(roi_result["logs"])
    if not roi_result["success"]:
        return {"success": False, "data": {}, "images": {**hough_result["images"], **roi_result["images"]}, "logs": logs}

    template_norm = template_data["signature_norm"]
    invert_angle = bool(radial_params.get("invert_angle", False))
    results = []
    roi_debug = {}
    tab_debug = {}
    for roi_item in roi_result["data"]["effective_rois"]:
        match_result = _match_roi_for_pipeline(roi_item, template_norm, tab_edge_params, radial_params, invert_angle)
        status = "ok" if match_result["success"] else "fail"
        results.append(
            {
                "id": roi_item["id"],
                "center_x": float(roi_item.get("center_x", roi_item["circle"]["x"])),
                "center_y": float(roi_item.get("center_y", roi_item["circle"]["y"])),
                "radius": float(roi_item.get("radius_full", roi_item["radius"])),
                "angle_deg": float(match_result.get("angle_deg", 0.0)),
                "min_error": float(match_result.get("min_error", 0.0)),
                "status": status,
            }
        )
        roi_debug["roi_{:02d}".format(roi_item["id"])] = roi_item["roi"]
        if match_result.get("tab_edges_clean") is not None:
            tab_debug["tab_edges_{:02d}".format(roi_item["id"])] = match_result["tab_edges_clean"]

    final_result = draw_final_results(image, results)
    final_path = OUTPUT_DIR / "final_result.png"
    csv_path = OUTPUT_DIR / "results.csv"
    write_image(final_path, final_result)
    save_results_csv(csv_path, results)
    if save_debug:
        save_debug_images(OUTPUT_DIR / "roi_debug", roi_debug)
        save_debug_images(OUTPUT_DIR / "tab_edges_debug", tab_debug)
        save_debug_images(
            OUTPUT_DIR / "pipeline_debug",
            {"hough_filtered": hough_result["images"]["hough_filtered"], "final_result": final_result},
        )

    elapsed = time.perf_counter() - start
    ok_count = sum(1 for row in results if row["status"] == "ok")
    logs.append("Hoan tat {} stator ({} ok) trong {:.2f}s".format(len(results), ok_count, elapsed))

    return {
        "success": True,
        "data": {
            "final_result_path": str(final_path),
            "results_csv_path": str(csv_path),
            "results": results,
            "rois": roi_result["data"]["effective_rois"],
            "roi_debug": roi_debug,
            "tab_edge_debug": tab_debug,
            "elapsed_sec": elapsed,
        },
        "images": {
            "final_result": final_result,
            "roi_overview": roi_result["images"]["overview"],
            "hough_filtered": hough_result["images"]["hough_filtered"],
            "original": image,
        },
        "logs": logs,
    }
