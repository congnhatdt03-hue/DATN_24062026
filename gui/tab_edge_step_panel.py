"""GUI panel for step 3: tab-edge filtering."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import DEFAULT_ROI_PARAMS, DEFAULT_TAB_EDGE_PARAMS, INPUT_DIR, ROI_PRESET_PATH, TAB_EDGE_PRESET_PATH
from src.io_utils import read_image
from src.pipeline_runner import run_step_roi_refine, run_step_tab_edges
from src.preset_store import load_preset, save_preset
from src.roi_extractor import find_roi_item

from .common_widgets import StepPanelBase
from .preset_dialogs import ask_load_preset_path, ask_save_preset_path


FIELD_SPECS = [
    {"path": "yolo.enabled", "label": "yolo_enabled", "type": "bool", "group": "YOLO"},
    {"path": "yolo.model_path", "label": "model_path", "type": "str", "group": "YOLO"},
    {"path": "yolo.conf_threshold", "label": "conf_threshold", "type": "float", "group": "YOLO", "min": 0.05, "max": 0.95},
    {"path": "yolo.box_padding_ratio", "label": "box_padding_ratio", "type": "float", "group": "YOLO", "min": 0.0, "max": 0.5},
    {"path": "yolo.box_padding_min_px", "label": "box_padding_min_px", "type": "int", "group": "YOLO", "min": 0, "max": 64},
    {"path": "preprocess.use_clahe", "label": "use_clahe", "type": "bool", "group": "Preprocess"},
    {"path": "preprocess.clahe_clip_limit", "label": "clahe_clip_limit", "type": "float", "group": "Preprocess", "min": 0.1, "max": 10.0},
    {"path": "preprocess.blur_method", "label": "blur_method", "type": "str", "group": "Preprocess"},
    {"path": "preprocess.gaussian_kernel", "label": "gaussian_kernel", "type": "int", "group": "Preprocess", "min": 1, "max": 31},
    {"path": "preprocess.median_kernel", "label": "median_kernel", "type": "int", "group": "Preprocess", "min": 1, "max": 31},
    {"path": "canny.threshold1", "label": "canny_threshold1", "type": "int", "group": "Canny", "min": 0, "max": 500},
    {"path": "canny.threshold2", "label": "canny_threshold2", "type": "int", "group": "Canny", "min": 0, "max": 500},
    {"path": "contour_filter.min_area", "label": "min_area", "type": "int", "group": "Contour", "min": 0, "max": 5000},
    {"path": "contour_filter.min_area_ratio", "label": "min_area_ratio", "type": "float", "group": "Contour", "min": 0.0, "max": 0.02},
    {"path": "contour_filter.min_keep_distance_ratio", "label": "min_keep_dist_ratio", "type": "float", "group": "Contour", "min": 0.0, "max": 1.5},
    {"path": "contour_filter.outer_profile_bin_deg", "label": "outer_bin_deg", "type": "float", "group": "Contour", "min": 0.5, "max": 10.0},
    {"path": "contour_filter.max_point_gap_px", "label": "max_point_gap_px", "type": "float", "group": "Contour", "min": 1.0, "max": 100.0},
    {"path": "contour_filter.max_angle_gap_deg", "label": "max_angle_gap_deg", "type": "float", "group": "Contour", "min": 1.0, "max": 45.0},
    {"path": "contour_filter.radial_angle_tolerance_deg", "label": "radial_tol_deg", "type": "float", "group": "Contour", "min": 1.0, "max": 45.0},
    {"path": "morphology.use_close", "label": "use_close", "type": "bool", "group": "Morphology"},
    {"path": "morphology.close_kernel", "label": "close_kernel", "type": "int", "group": "Morphology", "min": 1, "max": 21},
    {"path": "morphology.close_iter", "label": "close_iter", "type": "int", "group": "Morphology", "min": 1, "max": 5},
]


class TabEdgeStepPanel(StepPanelBase):
    """Panel for tab-edge tuning."""

    def __init__(self, master, app):
        self.params = load_preset(TAB_EDGE_PRESET_PATH, DEFAULT_TAB_EDGE_PARAMS)
        super().__init__(master, app, FIELD_SPECS, self.params)
        self.roi_id_var = tk.StringVar()
        self.roi_path_var = tk.StringVar()
        ttk.Label(self.toolbar, text="ROI file").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.roi_path_var, width=28).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.toolbar, text="Chon ROI", command=self.choose_roi).pack(side="left", padx=3)
        ttk.Label(self.toolbar, text="ROI ID").pack(side="left", padx=(8, 2))
        self.roi_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_id_var, state="readonly", width=8)
        self.roi_combo.pack(side="left")
        ttk.Button(self.toolbar, text="Run", command=self.run_step).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save Preset", command=self.save_preset).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load Preset", command=self.load_preset_file).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save As...", command=self.save_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load As...", command=self.load_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Reset", command=self.reset_params).pack(side="left", padx=3)
        self.refresh_roi_ids()

    def refresh_roi_ids(self):
        roi_result = self.app.shared.get("roi_result")
        ids = [str(item["id"]) for item in roi_result["data"]["rois"]] if roi_result and roi_result["success"] else []
        self.roi_combo["values"] = ids
        if ids and not self.roi_id_var.get():
            self.roi_id_var.set(ids[0])

    def choose_roi(self):
        path = filedialog.askopenfilename(title="Chon ROI", initialdir=str(INPUT_DIR))
        if path:
            self.roi_path_var.set(path)

    def _store_shared_params(self):
        self.app.shared["tab_edge_params"] = self.params

    def save_preset(self):
        self.params = self.parameter_panel.get_data()
        save_preset(TAB_EDGE_PRESET_PATH, self.params)
        self._store_shared_params()
        messagebox.showinfo("Preset", "Da luu tab_edge_preset.json")

    def load_preset_file(self):
        self.params = load_preset(TAB_EDGE_PRESET_PATH, DEFAULT_TAB_EDGE_PARAMS)
        self.parameter_panel.set_data(self.params)
        self._store_shared_params()

    def save_preset_as(self):
        target_path = ask_save_preset_path(TAB_EDGE_PRESET_PATH, "Luu Tab Edge preset thanh file rieng")
        if not target_path:
            return
        self.params = self.parameter_panel.get_data()
        save_preset(target_path, self.params)
        self._store_shared_params()
        messagebox.showinfo(
            "Preset",
            "Da luu preset test tai:\n{}\n\nPreset goc trong thu muc presets khong bi thay doi.".format(target_path),
        )

    def load_preset_as(self):
        target_path = ask_load_preset_path(TAB_EDGE_PRESET_PATH, "Nap Tab Edge preset tu file rieng")
        if not target_path:
            return
        self.params = load_preset(target_path, DEFAULT_TAB_EDGE_PARAMS)
        self.parameter_panel.set_data(self.params)
        self._store_shared_params()
        messagebox.showinfo(
            "Preset",
            "Da nap preset test tu:\n{}\n\nPreset goc trong thu muc presets khong bi ghi de.".format(target_path),
        )

    def reset_params(self):
        self.parameter_panel.set_data(DEFAULT_TAB_EDGE_PARAMS)
        self.params = self.parameter_panel.get_data()
        self._store_shared_params()

    def _manual_roi_item(self, path):
        image = read_image(path)
        if image is None:
            raise ValueError("Khong doc duoc ROI thu cong.")
        h, w = image.shape[:2]
        return {
            "id": 1,
            "roi": image,
            "offset_x": 0,
            "offset_y": 0,
            "center_in_roi": (w / 2.0, h / 2.0),
            "radius": min(w, h) * 0.35,
            "circle": {"id": 1, "x": w / 2.0, "y": h / 2.0, "r": min(w, h) * 0.35},
        }

    def _resolve_roi_item(self):
        roi_path = self.roi_path_var.get().strip()
        if roi_path:
            return self._manual_roi_item(roi_path)
        roi_result = self.app.shared.get("roi_result")
        if not roi_result or not roi_result["success"]:
            raise ValueError("Chua co ROI tu buoc 2.")
        roi_id = self.roi_id_var.get().strip() or "1"
        effective_item = self.app.shared.get("roi_effective_items", {}).get(int(roi_id))
        if effective_item is not None:
            return effective_item
        base_item = find_roi_item(roi_result["data"]["rois"], roi_id)
        if base_item is not None:
            roi_params = self.app.shared.get("roi_params") or load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS)
            refine_result = run_step_roi_refine(base_item, roi_params)
            if refine_result["success"]:
                refined_item = refine_result["data"]["roi_item"]
                self.app.shared.setdefault("roi_effective_items", {})[refined_item["id"]] = refined_item
                return refined_item
            return base_item
        raise ValueError("Khong tim thay ROI ID da chon.")

    def run_step(self):
        self.refresh_roi_ids()
        try:
            roi_item = self._resolve_roi_item()
        except Exception as exc:
            messagebox.showwarning("ROI", str(exc))
            return
        self.params = self.parameter_panel.get_data()
        result = run_step_tab_edges(roi_item, self.params)
        if result["success"]:
            self.app.shared.setdefault("tab_edge_results", {})[roi_item["id"]] = result
            self.app.shared.setdefault("tab_edge_roi_items", {})[roi_item["id"]] = roi_item
            self.app.shared["tab_edge_params"] = self.params
        self.set_result(result)
