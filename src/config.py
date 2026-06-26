"""Project configuration and default parameters."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
TEMPLATE_DIR = DATA_DIR / "template"
ROI_DIR = DATA_DIR / "roi"
OUTPUT_DIR = DATA_DIR / "output"
PRESET_DIR = BASE_DIR / "presets"
TAB_EDGE_YOLO_MODEL_PATH = BASE_DIR / "best.pt"

HOUGH_PRESET_PATH = PRESET_DIR / "hough_preset.json"
ROI_PRESET_PATH = PRESET_DIR / "roi_preset.json"
TAB_EDGE_PRESET_PATH = PRESET_DIR / "tab_edge_preset.json"
RADIAL_PRESET_PATH = PRESET_DIR / "radial_preset.json"
RADIAL_SIGNATURE_PRESET_PATH = PRESET_DIR / "radial_signature_preset.json"
TEMPLATE_DATA_PATH = PRESET_DIR / "template_data.json"
TEMPLATE_ROI_PATH = PRESET_DIR / "template_roi.png"

DEFAULT_HOUGH_PARAMS = {
    "expected_count": 12,
    "fast_mode": {
        "enabled": False,
        "max_processing_dim": 1400,
    },
    "preprocess": {
        "enabled": True,
        "use_clahe": True,
        "clahe_clip_limit": 2.5,
        "clahe_tile_grid_size": 8,
        "use_gaussian": True,
        "gaussian_kernel": 5,
    },
    "hough": {
        "dp": 1.2,
        "param1": 110,
        "param2": 38,
        "minDist": 120,
        "minRadius": 32,
        "maxRadius": 140,
        "min_center_dist": 90,
    },
    "filter": {
        "use_radius_consensus": True,
        "radius_consensus_tol": 6,
        "force_common_radius": True,
    },
}

DEFAULT_ROI_PARAMS = {
    "half_size_scale": 1.30,
    "output_size": 0,
    "refine": {
        "enabled": True,
        "preprocess": {
            "use_clahe": True,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": 8,
            "use_gaussian": True,
            "gaussian_kernel": 5,
        },
        "canny": {
            "threshold1": 70,
            "threshold2": 170,
        },
        "mask": {
            "enabled": True,
            "inner_radius_scale": 0.82,
            "outer_radius_scale": 1.20,
        },
        "hough": {
            "dp": 1.2,
            "param1": 110,
            "param2": 30,
            "minDist": 20,
            "min_radius_scale": 0.78,
            "max_radius_scale": 1.18,
            "max_center_shift_scale": 0.45,
        },
        "least_squares": {
            "enabled": False,
            "band_width_px": 3.0,
            "min_points": 24,
            "max_center_shift_scale": 0.12,
            "max_radius_delta_scale": 0.12,
            "score_tolerance": 0.02,
        },
        "score": {
            "ring_width": 3,
            "center_penalty_weight": 0.10,
            "radius_penalty_weight": 0.20,
        },
    },
}

DEFAULT_TAB_EDGE_PARAMS = {
    "yolo": {
        "enabled": True,
        "model_path": str(TAB_EDGE_YOLO_MODEL_PATH),
        "conf_threshold": 0.25,
        "box_padding_ratio": 0.10,
        "box_padding_min_px": 12,
    },
    "preprocess": {
        "use_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": 8,
        "blur_method": "gaussian",
        "gaussian_kernel": 5,
        "median_kernel": 5,
        "bilateral_d": 7,
        "bilateral_sigma_color": 50,
        "bilateral_sigma_space": 50,
    },
    "canny": {
        "threshold1": 70,
        "threshold2": 170,
        "aperture_size": 3,
        "l2_gradient": False,
    },
    "contour_filter": {
        "min_area": 24,
        "min_area_ratio": 0.0015,
        "min_keep_distance_ratio": 0.88,
        "outer_profile_bin_deg": 1.0,
        "max_point_gap_px": 28.0,
        "max_angle_gap_deg": 5.0,
        "radial_angle_tolerance_deg": 18.0,
    },
    "morphology": {
        "use_close": True,
        "close_kernel": 3,
        "close_iter": 1,
        "use_dilate": False,
        "dilate_kernel": 3,
        "dilate_iter": 1,
    },
}

DEFAULT_RADIAL_PARAMS = {
    "source_mode": "closed_edges",
    "use_radius_band": True,
    "inner_radius_scale": 1.03,
    "outer_radius_scale": 1.34,
    "use_source_dilate": True,
    "source_dilate_kernel": 3,
    "source_dilate_iter": 1,
    "num_angles": 360,
    "ray_step_px": 1.0,
    "ray_thickness": 2,
    "min_valid_radius_scale": 1.0,
    "floor_to_radius": True,
    "reject_outliers": True,
    "outlier_window": 11,
    "outlier_max_delta": 16.0,
    "interpolate_missing": True,
    "max_gap_to_interpolate": 10,
    "smooth_signature": True,
    "smooth_window": 7,
    "scale_normalize": True,
    "invert_angle": False,
}
