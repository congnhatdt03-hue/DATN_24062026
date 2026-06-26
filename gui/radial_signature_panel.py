"""Combined GUI panel for steps 3 and 4: Tab Edges + Radial Signature."""

import copy
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import (
    DEFAULT_RADIAL_PARAMS,
    DEFAULT_ROI_PARAMS,
    DEFAULT_TAB_EDGE_PARAMS,
    INPUT_DIR,
    RADIAL_SIGNATURE_PRESET_PATH,
    ROI_PRESET_PATH,
)
from src.io_utils import read_image
from src.pipeline_runner import run_step_radial, run_step_roi_refine, run_step_tab_edges
from src.preset_store import load_preset, load_radial_signature_preset, save_radial_signature_preset
from src.roi_extractor import find_roi_item

from .common_widgets import StepPanelBase, get_nested, set_nested
from .preset_dialogs import ask_load_preset_path, ask_save_preset_path
from .radial_step_panel import FIELD_SPECS as RADIAL_FIELD_SPECS
from .tab_edge_step_panel import FIELD_SPECS as TAB_EDGE_FIELD_SPECS


def _clone_specs_with_group_prefix(specs, prefix):
    cloned = []
    for spec in specs:
        item = copy.deepcopy(spec)
        item["group"] = "{} / {}".format(prefix, spec.get("group", "General"))
        cloned.append(item)
    return cloned


FIELD_SPECS = _clone_specs_with_group_prefix(TAB_EDGE_FIELD_SPECS, "Tab Edges") + _clone_specs_with_group_prefix(
    RADIAL_FIELD_SPECS, "Radial"
)
TAB_EDGE_PATHS = [spec["path"] for spec in TAB_EDGE_FIELD_SPECS]
RADIAL_PATHS = [spec["path"] for spec in RADIAL_FIELD_SPECS]


class RadialSignaturePanel(StepPanelBase):
    """One panel that lets the user tune tab-edge filtering and radial signature together."""

    def __init__(self, master, app):
        preset_bundle = load_radial_signature_preset(RADIAL_SIGNATURE_PRESET_PATH)
        self.tab_edge_params = preset_bundle["tab_edge_params"]
        self.radial_params = preset_bundle["radial_params"]
        self.roi_id_var = tk.StringVar()
        self.roi_path_var = tk.StringVar()
        self._last_run_mode = "radial"
        super().__init__(master, app, FIELD_SPECS, self._combine_panel_data())
        ttk.Label(self.toolbar, text="ROI file").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.roi_path_var, width=28).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.toolbar, text="Chon ROI", command=self.choose_roi).pack(side="left", padx=3)
        ttk.Label(self.toolbar, text="ROI ID").pack(side="left", padx=(8, 2))
        self.roi_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_id_var, state="readonly", width=8)
        self.roi_combo.pack(side="left")
        ttk.Button(self.toolbar, text="Run Tab Edges", command=self.run_tab_edges).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Run Radial", command=self.run_radial).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save Preset", command=self.save_preset).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load Preset", command=self.load_preset_file).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save As...", command=self.save_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load As...", command=self.load_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Reset", command=self.reset_params).pack(side="left", padx=3)
        self.refresh_roi_ids()
        self._store_shared_params()

    def _combine_panel_data(self):
        combined = copy.deepcopy(self.tab_edge_params)
        for path in RADIAL_PATHS:
            set_nested(combined, path, copy.deepcopy(get_nested(self.radial_params, path)))
        return combined

    def _split_panel_data(self, panel_data):
        tab_edge_params = {}
        radial_params = {}
        for path in TAB_EDGE_PATHS:
            set_nested(tab_edge_params, path, copy.deepcopy(get_nested(panel_data, path)))
        for path in RADIAL_PATHS:
            set_nested(radial_params, path, copy.deepcopy(get_nested(panel_data, path)))
        return tab_edge_params, radial_params

    def _capture_params(self):
        panel_data = self.parameter_panel.get_data()
        self.tab_edge_params, self.radial_params = self._split_panel_data(panel_data)
        self._store_shared_params()

    def _store_shared_params(self):
        self.app.shared["tab_edge_params"] = copy.deepcopy(self.tab_edge_params)
        self.app.shared["radial_params"] = copy.deepcopy(self.radial_params)

    def refresh_roi_ids(self):
        roi_result = self.app.shared.get("roi_result")
        ids = [str(item["id"]) for item in roi_result["data"]["rois"]] if roi_result and roi_result["success"] else []
        self.roi_combo["values"] = ids
        if ids and self.roi_id_var.get() not in ids:
            self.roi_id_var.set(ids[0])

    def choose_roi(self):
        path = filedialog.askopenfilename(title="Chon ROI", initialdir=str(INPUT_DIR))
        if path:
            self.roi_path_var.set(path)

    def save_preset(self):
        self._capture_params()
        save_radial_signature_preset(RADIAL_SIGNATURE_PRESET_PATH, self.tab_edge_params, self.radial_params)
        messagebox.showinfo("Preset", "Da luu radial_signature_preset.json")

    def load_preset_file(self):
        preset_bundle = load_radial_signature_preset(RADIAL_SIGNATURE_PRESET_PATH)
        self.tab_edge_params = preset_bundle["tab_edge_params"]
        self.radial_params = preset_bundle["radial_params"]
        self.parameter_panel.set_data(self._combine_panel_data())
        self._store_shared_params()

    def save_preset_as(self):
        target_path = ask_save_preset_path(RADIAL_SIGNATURE_PRESET_PATH, "Luu preset Tab Edges + Radial thanh file rieng")
        if not target_path:
            return
        self._capture_params()
        save_radial_signature_preset(target_path, self.tab_edge_params, self.radial_params)
        messagebox.showinfo(
            "Preset",
            "Da luu bundle preset test tai:\n{}\n\nPreset goc trong thu muc presets khong bi thay doi.".format(target_path),
        )

    def load_preset_as(self):
        target_path = ask_load_preset_path(RADIAL_SIGNATURE_PRESET_PATH, "Nap preset Tab Edges + Radial tu file rieng")
        if not target_path:
            return
        preset_bundle = load_radial_signature_preset(target_path)
        self.tab_edge_params = preset_bundle["tab_edge_params"]
        self.radial_params = preset_bundle["radial_params"]
        self.parameter_panel.set_data(self._combine_panel_data())
        self._store_shared_params()
        messagebox.showinfo(
            "Preset",
            "Da nap bundle preset test tu:\n{}\n\nPreset goc trong thu muc presets khong bi ghi de.".format(target_path),
        )

    def reset_params(self):
        self.tab_edge_params = copy.deepcopy(DEFAULT_TAB_EDGE_PARAMS)
        self.radial_params = copy.deepcopy(DEFAULT_RADIAL_PARAMS)
        self.parameter_panel.set_data(self._combine_panel_data())
        self._store_shared_params()

    def _manual_roi_item(self, path):
        image = read_image(path)
        if image is None:
            raise ValueError("Khong doc duoc ROI thu cong.")
        height, width = image.shape[:2]
        radius = min(width, height) * 0.35
        return {
            "id": 1,
            "roi": image,
            "offset_x": 0,
            "offset_y": 0,
            "center_in_roi": (width / 2.0, height / 2.0),
            "radius": radius,
            "circle": {"id": 1, "x": width / 2.0, "y": height / 2.0, "r": radius},
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

    def _store_step_results(self, roi_item, tab_result, radial_result=None):
        if tab_result and tab_result.get("success"):
            self.app.shared.setdefault("tab_edge_results", {})[roi_item["id"]] = tab_result
            self.app.shared.setdefault("tab_edge_roi_items", {})[roi_item["id"]] = roi_item
        if radial_result and radial_result.get("success"):
            self.app.shared.setdefault("radial_results", {})[roi_item["id"]] = radial_result

    def run_tab_edges(self):
        self._run_mode("tab_edge")

    def run_radial(self):
        self._run_mode("radial")

    def _run_mode(self, mode):
        self.refresh_roi_ids()
        try:
            roi_item = self._resolve_roi_item()
        except Exception as exc:
            messagebox.showwarning("ROI", str(exc))
            return
        self._capture_params()
        self._last_run_mode = mode

        tab_result = run_step_tab_edges(roi_item, self.tab_edge_params)
        self._store_step_results(roi_item, tab_result)
        if not tab_result["success"] or mode == "tab_edge":
            self.set_result(tab_result)
            return

        radial_result = run_step_radial(roi_item, tab_result["images"], self.radial_params)
        self._store_step_results(roi_item, tab_result, radial_result)
        if radial_result["success"]:
            combined_result = {
                "success": True,
                "data": {
                    **radial_result.get("data", {}),
                    "roi_id": roi_item.get("id"),
                },
                "images": radial_result.get("images", {}),
                "logs": list(tab_result.get("logs", [])) + list(radial_result.get("logs", [])),
            }
        else:
            combined_result = {
                "success": False,
                "data": {},
                "images": radial_result.get("images", {}) or tab_result.get("images", {}),
                "logs": list(tab_result.get("logs", [])) + list(radial_result.get("logs", [])),
            }
        self.set_result(combined_result)

    def run_step(self):
        if self._last_run_mode == "tab_edge":
            self.run_tab_edges()
            return
        self.run_radial()
