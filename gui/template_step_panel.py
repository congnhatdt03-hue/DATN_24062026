"""GUI panel for step 5: template creation."""

import copy
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import (
    DEFAULT_ROI_PARAMS,
    INPUT_DIR,
    ROI_PRESET_PATH,
    TEMPLATE_DATA_PATH,
    TEMPLATE_DIR,
    TEMPLATE_ROI_PATH,
)
from src.image_transform import rotate_image_keep_size
from src.io_utils import read_image, write_image
from src.pipeline_runner import run_step_template
from src.preset_store import load_preset, load_radial_signature_preset
from src.roi_extractor import build_roi_item_from_image
from src.template_builder import save_template_bundle
from src.visualization import draw_center_axes_overlay

from .common_widgets import StepPanelBase
from .preset_dialogs import ask_save_image_path, ask_save_json_path


class TemplateStepPanel(StepPanelBase):
    """Panel for creating template_data.json from a detected stator ID or an external ROI."""

    def __init__(self, master, app):
        super().__init__(master, app, [], {})
        self.roi_path_var = tk.StringVar()
        self.roi_id_var = tk.StringVar()
        self.rotation_angle_var = tk.StringVar(value="0.00")
        self.rotation_step_var = tk.StringVar(value="1.00")
        self._rotation_sync_guard = False
        self._rotation_preview_job = None
        ttk.Label(self.toolbar, text="Stator ID").pack(side="left")
        self.roi_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_id_var, state="readonly", width=8)
        self.roi_combo.pack(side="left", padx=(2, 8))
        ttk.Label(self.toolbar, text="hoac file").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.roi_path_var, width=20).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.toolbar, text="Chon ROI", command=self.choose_roi).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Run", command=self.run_step).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save Template", command=self.save_template).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save As Template...", command=self.save_template_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save Image...", command=self.save_current_debug_image).pack(side="left", padx=3)
        self.latest_result = None
        self.latest_roi_item = None
        self._build_rotation_controls()
        self.refresh_ids()

    def _build_rotation_controls(self):
        ttk.Label(self.toolbar, text="Xoay").pack(side="left", padx=(10, 4))
        self.rotation_scale = ttk.Scale(
            self.toolbar,
            orient="horizontal",
            from_=-180.0,
            to=180.0,
            length=170,
            command=self._on_rotation_scale,
        )
        self.rotation_scale.pack(side="left", padx=(0, 4))
        self.rotation_scale.set(0.0)

        self.rotation_entry = ttk.Entry(self.toolbar, textvariable=self.rotation_angle_var, width=7)
        self.rotation_entry.pack(side="left", padx=(0, 6))
        self.rotation_entry.bind("<Return>", lambda _event: self._commit_rotation_entry())
        self.rotation_entry.bind("<FocusOut>", lambda _event: self._commit_rotation_entry())

        ttk.Label(self.toolbar, text="Buoc").pack(side="left", padx=(2, 4))
        self.rotation_step_entry = ttk.Entry(self.toolbar, textvariable=self.rotation_step_var, width=6)
        self.rotation_step_entry.pack(side="left", padx=(0, 4))
        self.rotation_step_entry.bind("<Return>", lambda _event: self._commit_rotation_step_entry())
        self.rotation_step_entry.bind("<FocusOut>", lambda _event: self._commit_rotation_step_entry())

        ttk.Button(self.toolbar, text="<-", command=lambda: self._nudge_rotation(-1.0)).pack(side="left", padx=2)
        ttk.Button(self.toolbar, text="->", command=lambda: self._nudge_rotation(1.0)).pack(side="left", padx=2)

    def refresh_ids(self):
        """Populate the stator ID combobox from the ROI step result."""
        roi_result = self.app.shared.get("roi_result")
        ids = [str(item["id"]) for item in roi_result["data"]["rois"]] if roi_result and roi_result["success"] else []
        self.roi_combo["values"] = ids
        if ids and not self.roi_id_var.get():
            self.roi_id_var.set(ids[0])

    def choose_roi(self):
        path = filedialog.askopenfilename(title="Chon ROI mau 0 do", initialdir=str(INPUT_DIR))
        if path:
            self.roi_path_var.set(path)

    @staticmethod
    def _normalize_rotation(angle_deg):
        """Keep the user angle in [-180, 180] with clockwise positive."""
        angle = float(angle_deg)
        normalized = ((angle + 180.0) % 360.0) - 180.0
        if abs(normalized + 180.0) < 1e-9 and angle > 0:
            normalized = 180.0
        return round(normalized, 2)

    @staticmethod
    def _safe_float(text, fallback):
        try:
            return float(text)
        except (TypeError, ValueError):
            return float(fallback)

    def _current_rotation_angle(self):
        return self._safe_float(self.rotation_angle_var.get(), 0.0)

    def _set_rotation_angle(self, angle_deg, trigger_preview=True):
        angle = self._normalize_rotation(angle_deg)
        text = "{:.2f}".format(angle)
        if self.rotation_angle_var.get() == text and abs(float(self.rotation_scale.get()) - angle) < 1e-6:
            if trigger_preview:
                self._schedule_rotation_preview()
            return
        self._rotation_sync_guard = True
        try:
            self.rotation_angle_var.set(text)
            self.rotation_scale.set(angle)
        finally:
            self._rotation_sync_guard = False
        if trigger_preview:
            self._schedule_rotation_preview()

    def _on_rotation_scale(self, raw_value):
        if self._rotation_sync_guard:
            return
        angle = self._safe_float(raw_value, 0.0)
        self._set_rotation_angle(angle, trigger_preview=True)

    def _commit_rotation_entry(self):
        self._set_rotation_angle(self._current_rotation_angle(), trigger_preview=True)

    def _commit_rotation_step_entry(self):
        step = abs(self._safe_float(self.rotation_step_var.get(), 1.0))
        if step < 0.01:
            step = 0.01
        self.rotation_step_var.set("{:.2f}".format(step))

    def _nudge_rotation(self, direction):
        self._commit_rotation_step_entry()
        step = self._safe_float(self.rotation_step_var.get(), 1.0)
        self._set_rotation_angle(self._current_rotation_angle() + (float(direction) * step), trigger_preview=True)

    def _has_rotation_source(self):
        roi_path = self.roi_path_var.get().strip()
        if roi_path:
            return Path(roi_path).is_file()
        roi_result = self.app.shared.get("roi_result")
        if roi_result and roi_result.get("success") and roi_result.get("data", {}).get("rois"):
            return True
        effective_items = self.app.shared.get("roi_effective_items", {})
        return bool(effective_items)

    def _schedule_rotation_preview(self):
        if self._rotation_preview_job is not None:
            self.after_cancel(self._rotation_preview_job)
            self._rotation_preview_job = None
        if not self._has_rotation_source():
            return
        if self.latest_result is None and not self.current_images:
            return
        self._rotation_preview_job = self.after(180, self._run_rotation_preview)

    def _run_rotation_preview(self):
        self._rotation_preview_job = None
        self.run_step()

    def _ensure_rotation_result_current(self):
        """Refresh tab 5 if the cached result does not match the current angle."""
        if not self._has_rotation_source():
            return
        if self._rotation_preview_job is not None:
            self.after_cancel(self._rotation_preview_job)
            self._rotation_preview_job = None
        latest_angle = None
        if self.latest_result is not None:
            latest_angle = self._safe_float(self.latest_result.get("data", {}).get("rotation_deg", 0.0), 0.0)
        if self.latest_result is None or latest_angle is None or abs(latest_angle - self._current_rotation_angle()) > 0.005:
            self.run_step()

    def _roi_item_by_id(self, roi_id):
        """Return the best available ROI item for one detected stator ID."""
        effective_items = self.app.shared.get("roi_effective_items", {})
        if roi_id in effective_items:
            return effective_items[roi_id]
        roi_result = self.app.shared.get("roi_result")
        if roi_result and roi_result["success"]:
            for item in roi_result["data"]["rois"]:
                if item["id"] == roi_id:
                    return item
        return None

    def _resolve_roi_item(self):
        """Pick the ROI item: external file first, then selected ID, then first ROI."""
        roi_path = self.roi_path_var.get().strip()
        if roi_path:
            image = read_image(roi_path)
            if image is None:
                raise ValueError("Khong doc duoc ROI mau.")
            roi_params = self.app.shared.get("roi_params") or load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS)
            return build_roi_item_from_image(image, roi_params, roi_id=1)
        roi_id_text = self.roi_id_var.get().strip()
        if roi_id_text:
            item = self._roi_item_by_id(int(roi_id_text))
            if item is not None:
                return item
        effective_items = self.app.shared.get("roi_effective_items", {})
        if effective_items:
            return effective_items[sorted(effective_items.keys())[0]]
        roi_result = self.app.shared.get("roi_result")
        if roi_result and roi_result["success"] and roi_result["data"]["rois"]:
            return roi_result["data"]["rois"][0]
        raise ValueError("Hay chon stator ID (chay buoc 2 truoc) hoac chon file ROI mau.")

    def _apply_rotation_to_roi_item(self, roi_item):
        angle_deg = self._current_rotation_angle()
        if abs(angle_deg) < 1e-9:
            return roi_item
        rotated_item = copy.deepcopy(roi_item)
        rotated_item["roi"] = rotate_image_keep_size(
            roi_item["roi"],
            angle_deg=angle_deg,
            center=roi_item.get("center_in_roi"),
        )
        logs = list(rotated_item.get("logs", []))
        logs.append(
            "Template rotate: {:+.2f} deg (duong la xoay cung chieu kim dong ho)".format(angle_deg)
        )
        rotated_item["logs"] = logs
        return rotated_item

    @staticmethod
    def _full_center(roi_item):
        center_in_roi = roi_item.get("center_in_roi", (0.0, 0.0))
        return [
            float(roi_item.get("center_x", center_in_roi[0])),
            float(roi_item.get("center_y", center_in_roi[1])),
        ]

    def run_step(self):
        self.refresh_ids()
        try:
            roi_item = self._apply_rotation_to_roi_item(self._resolve_roi_item())
        except Exception as exc:
            messagebox.showwarning("Template", str(exc))
            return
        preset_bundle = load_radial_signature_preset()
        tab_params = self.app.shared.get("tab_edge_params") or preset_bundle["tab_edge_params"]
        radial_params = self.app.shared.get("radial_params") or preset_bundle["radial_params"]
        result = run_step_template(roi_item, tab_params, radial_params)
        roi_logs = roi_item.get("logs") or []
        if roi_logs:
            result["logs"] = list(roi_logs) + list(result.get("logs", []))
        result.setdefault("images", {})
        result["images"]["roi_with_axes"] = draw_center_axes_overlay(roi_item["roi"])
        self.latest_result = result
        self.latest_roi_item = roi_item
        if result["success"]:
            # Gan kem metadata stator that (ID, tam toan anh, ban kinh) de buoc Match hien thi.
            result["data"]["source_id"] = int(roi_item.get("id", 1))
            result["data"]["center_full"] = self._full_center(roi_item)
            result["data"]["radius_full"] = float(roi_item.get("radius_full", roi_item.get("radius", 0.0)))
            result["data"]["rotation_deg"] = self._current_rotation_angle()
            self.app.shared["template_data"] = result["data"]
            self.app.shared["template_roi_image"] = result["images"].get("template_roi")
        self.set_result(result)

    def _ensure_template_ready(self):
        self._ensure_rotation_result_current()
        if not self.latest_result or not self.latest_result["success"]:
            messagebox.showwarning("Template", "Hay run template truoc.")
            return None, None
        return self.latest_result["data"], self.latest_result["images"].get("template_roi")

    def save_template(self):
        template_data, template_roi = self._ensure_template_ready()
        if template_data is None:
            return
        saved_payload, _json_path, _roi_path = save_template_bundle(
            TEMPLATE_DATA_PATH,
            template_data,
            template_roi_image=template_roi,
            template_roi_path=TEMPLATE_ROI_PATH,
        )
        self.latest_result["data"] = saved_payload
        self.app.shared["template_data"] = saved_payload
        messagebox.showinfo("Template", "Da luu template_data.json va template_roi.png")

    def save_template_as(self):
        template_data, template_roi = self._ensure_template_ready()
        if template_data is None:
            return
        target_path = ask_save_json_path(TEMPLATE_DATA_PATH, "Luu template ra file ngoai")
        if not target_path:
            return
        _saved_payload, json_path, roi_path = save_template_bundle(
            target_path,
            template_data,
            template_roi_image=template_roi,
            store_relative_roi_path=True,
        )
        roi_name = roi_path.name if roi_path is not None else "(khong co anh ROI)"
        message = (
            "Da luu template test tai:\n{}\n\nAnh ROI di kem: {}\n"
            "Nut Save Template mac dinh van luu o thu muc presets."
        ).format(
            json_path,
            roi_name,
        )
        messagebox.showinfo("Template", message)

    def save_current_debug_image(self):
        self._ensure_rotation_result_current()
        if not self.current_images:
            messagebox.showwarning("Save Image", "Chua co anh debug de luu. Hay run tab 5 truoc.")
            return
        selected_name = self.debug_selector.var.get().strip()
        image = self.current_images.get(selected_name)
        if image is None and self.current_images:
            selected_name, image = next(iter(self.current_images.items()))
        if image is None:
            messagebox.showwarning("Save Image", "Anh debug dang xem hien khong hop le.")
            return

        angle_text = "{:+.2f}".format(self._current_rotation_angle()).replace("+", "cw_").replace("-", "ccw_").replace(".", "p")
        default_name = "{}_{}.png".format(selected_name or "template_debug", angle_text)
        target_path = ask_save_image_path(
            initialdir=TEMPLATE_DIR,
            initialfile=default_name,
            title="Luu anh debug dang xem",
        )
        if not target_path:
            return
        write_image(target_path, image)
        messagebox.showinfo("Save Image", "Da luu anh debug tai:\n{}".format(target_path))
