"""GUI panel for step 4: radial signature."""

import tkinter as tk
from tkinter import messagebox, ttk

from src.config import DEFAULT_RADIAL_PARAMS, RADIAL_PRESET_PATH
from src.pipeline_runner import run_step_radial
from src.preset_store import load_preset, save_preset

from .common_widgets import StepPanelBase
from .preset_dialogs import ask_load_preset_path, ask_save_preset_path


FIELD_SPECS = [
    {"path": "source_mode", "label": "source_mode", "type": "str", "group": "Source"},
    {"path": "use_radius_band", "label": "use_radius_band", "type": "bool", "group": "Source"},
    {"path": "inner_radius_scale", "label": "inner_radius_scale", "type": "float", "group": "Source", "min": 0.8, "max": 1.5},
    {"path": "outer_radius_scale", "label": "outer_radius_scale", "type": "float", "group": "Source", "min": 1.0, "max": 2.0},
    {"path": "use_source_dilate", "label": "use_source_dilate", "type": "bool", "group": "Source"},
    {"path": "source_dilate_kernel", "label": "source_dilate_kernel", "type": "int", "group": "Source", "min": 1, "max": 9},
    {"path": "source_dilate_iter", "label": "source_dilate_iter", "type": "int", "group": "Source", "min": 0, "max": 4},
    {"path": "num_angles", "label": "num_angles", "type": "int", "group": "Radial", "min": 36, "max": 1440},
    {"path": "ray_step_px", "label": "ray_step_px", "type": "float", "group": "Radial", "min": 0.5, "max": 5.0},
    {"path": "ray_thickness", "label": "ray_thickness", "type": "int", "group": "Radial", "min": 0, "max": 6},
    {"path": "min_valid_radius_scale", "label": "min_valid_radius_scale", "type": "float", "group": "Radial", "min": 0.9, "max": 1.2},
    {"path": "floor_to_radius", "label": "floor_to_radius", "type": "bool", "group": "Radial"},
    {"path": "reject_outliers", "label": "reject_outliers", "type": "bool", "group": "Radial"},
    {"path": "outlier_window", "label": "outlier_window", "type": "int", "group": "Radial", "min": 3, "max": 31},
    {"path": "outlier_max_delta", "label": "outlier_max_delta", "type": "float", "group": "Radial", "min": 2.0, "max": 80.0},
    {"path": "interpolate_missing", "label": "interpolate_missing", "type": "bool", "group": "Radial"},
    {"path": "max_gap_to_interpolate", "label": "max_gap_to_interpolate", "type": "int", "group": "Radial", "min": 0, "max": 90},
    {"path": "smooth_signature", "label": "smooth_signature", "type": "bool", "group": "Radial"},
    {"path": "smooth_window", "label": "smooth_window", "type": "int", "group": "Radial", "min": 1, "max": 51},
    {"path": "scale_normalize", "label": "scale_normalize (rho/R)", "type": "bool", "group": "Match"},
    {"path": "invert_angle", "label": "invert_angle", "type": "bool", "group": "Match"},
]


class RadialStepPanel(StepPanelBase):
    """Panel for radial signature preview."""

    def __init__(self, master, app):
        self.params = load_preset(RADIAL_PRESET_PATH, DEFAULT_RADIAL_PARAMS)
        super().__init__(master, app, FIELD_SPECS, self.params)
        self.roi_id_var = tk.StringVar()
        ttk.Label(self.toolbar, text="ROI ID").pack(side="left")
        self.roi_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_id_var, state="readonly", width=8)
        self.roi_combo.pack(side="left", padx=6)
        ttk.Button(self.toolbar, text="Run", command=self.run_step).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save Preset", command=self.save_preset).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load Preset", command=self.load_preset_file).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save As...", command=self.save_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load As...", command=self.load_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Reset", command=self.reset_params).pack(side="left", padx=3)
        self.refresh_ids()

    def refresh_ids(self):
        roi_result = self.app.shared.get("roi_result")
        ids = [str(item["id"]) for item in roi_result["data"]["rois"]] if roi_result and roi_result["success"] else []
        self.roi_combo["values"] = ids
        if ids and not self.roi_id_var.get():
            self.roi_id_var.set(ids[0])

    def _store_shared_params(self):
        self.app.shared["radial_params"] = self.params

    def save_preset(self):
        self.params = self.parameter_panel.get_data()
        save_preset(RADIAL_PRESET_PATH, self.params)
        self._store_shared_params()
        messagebox.showinfo("Preset", "Da luu radial_preset.json")

    def load_preset_file(self):
        self.params = load_preset(RADIAL_PRESET_PATH, DEFAULT_RADIAL_PARAMS)
        self.parameter_panel.set_data(self.params)
        self._store_shared_params()

    def save_preset_as(self):
        target_path = ask_save_preset_path(RADIAL_PRESET_PATH, "Luu Radial preset thanh file rieng")
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
        target_path = ask_load_preset_path(RADIAL_PRESET_PATH, "Nap Radial preset tu file rieng")
        if not target_path:
            return
        self.params = load_preset(target_path, DEFAULT_RADIAL_PARAMS)
        self.parameter_panel.set_data(self.params)
        self._store_shared_params()
        messagebox.showinfo(
            "Preset",
            "Da nap preset test tu:\n{}\n\nPreset goc trong thu muc presets khong bi ghi de.".format(target_path),
        )

    def reset_params(self):
        self.parameter_panel.set_data(DEFAULT_RADIAL_PARAMS)
        self.params = self.parameter_panel.get_data()
        self._store_shared_params()

    def run_step(self):
        self.refresh_ids()
        roi_result = self.app.shared.get("roi_result")
        if not roi_result or not roi_result["success"]:
            messagebox.showwarning("Radial", "Hay chay ROI truoc.")
            return
        roi_id = int(self.roi_id_var.get() or "1")
        roi_item = self.app.shared.get("tab_edge_roi_items", {}).get(roi_id)
        if roi_item is None:
            roi_item = self.app.shared.get("roi_effective_items", {}).get(roi_id)
        if roi_item is None:
            roi_item = next((item for item in roi_result["data"]["rois"] if item["id"] == roi_id), None)
        tab_result = self.app.shared.get("tab_edge_results", {}).get(roi_id)
        if roi_item is None or tab_result is None or not tab_result["success"]:
            messagebox.showwarning("Radial", "Hay chay Tab Edges cho ROI nay truoc.")
            return
        self.params = self.parameter_panel.get_data()
        result = run_step_radial(roi_item, tab_result["images"], self.params)
        if result["success"]:
            self.app.shared.setdefault("radial_results", {})[roi_id] = result
            self.app.shared["radial_params"] = self.params
        self.set_result(result)
