"""GUI panel for step 2: ROI extraction + local Hough refinement."""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.config import DEFAULT_HOUGH_PARAMS, DEFAULT_ROI_PARAMS, HOUGH_PRESET_PATH, INPUT_DIR, ROI_DIR, ROI_PRESET_PATH
from src.pipeline_runner import run_step_hough, run_step_roi_crop, run_step_roi_refine
from src.preset_store import load_preset, save_preset
from src.roi_extractor import find_roi_item

from .common_widgets import ResultTable, StepPanelBase
from .preset_dialogs import ask_load_preset_path, ask_save_preset_path


FIELD_SPECS = [
    {"path": "half_size_scale", "label": "half_size_scale", "type": "float", "group": "ROI", "min": 0.8, "max": 2.8},
    {"path": "output_size", "label": "output_size", "type": "int", "group": "ROI", "min": 0, "max": 1024},
    {"path": "refine.enabled", "label": "Enable refine", "type": "bool", "group": "Hough Refine"},
    {"path": "refine.least_squares.enabled", "label": "Use least-squares", "type": "bool", "group": "Hough Refine"},
    {"path": "refine.preprocess.use_clahe", "label": "Use CLAHE", "type": "bool", "group": "Preprocess"},
    {"path": "refine.preprocess.clahe_clip_limit", "label": "CLAHE clip", "type": "float", "group": "Preprocess", "min": 0.1, "max": 10.0},
    {"path": "refine.preprocess.clahe_tile_grid_size", "label": "CLAHE tile", "type": "int", "group": "Preprocess", "min": 1, "max": 32},
    {"path": "refine.preprocess.use_gaussian", "label": "Use Gaussian", "type": "bool", "group": "Preprocess"},
    {"path": "refine.preprocess.gaussian_kernel", "label": "Gaussian kernel", "type": "int", "group": "Preprocess", "min": 1, "max": 31},
    {"path": "refine.canny.threshold1", "label": "canny_threshold1", "type": "int", "group": "Canny", "min": 0, "max": 500},
    {"path": "refine.canny.threshold2", "label": "canny_threshold2", "type": "int", "group": "Canny", "min": 0, "max": 500},
    {"path": "refine.hough.dp", "label": "Hough dp", "type": "float", "group": "Hough Refine", "min": 1.0, "max": 3.0},
    {"path": "refine.hough.param1", "label": "Hough param1", "type": "float", "group": "Hough Refine", "min": 1, "max": 300},
    {"path": "refine.hough.param2", "label": "Hough param2", "type": "float", "group": "Hough Refine", "min": 1, "max": 200},
    {"path": "refine.hough.minDist", "label": "minDist", "type": "float", "group": "Hough Refine", "min": 1, "max": 300},
    {"path": "refine.hough.min_radius_scale", "label": "R min scale", "type": "float", "group": "Hough Refine", "min": 0.4, "max": 1.2},
    {"path": "refine.hough.max_radius_scale", "label": "R max scale", "type": "float", "group": "Hough Refine", "min": 0.8, "max": 1.8},
    {"path": "refine.hough.max_center_shift_scale", "label": "Center shift scale", "type": "float", "group": "Hough Refine", "min": 0.05, "max": 1.0},
]

POLL_MS = 60
DISPLAY_NAME_BY_KEY = {
    "overview": "Overview",
    "selected_roi": "Selected ROI",
    "selected_roi_detected": "Selected ROI Detected",
    "selected_roi_preprocessed": "Selected ROI Preprocessed",
    "selected_roi_edges": "Selected ROI Edges",
    "selected_roi_masked_edges": "Selected ROI Masked Edges",
}


class RoiStepPanel(StepPanelBase):
    """Panel for ROI extraction and local Hough refinement."""

    def __init__(self, master, app):
        self.params = load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS)
        self._busy = False
        self._pending = None
        self._event_queue = queue.Queue()
        self._poll_id = None
        self._source_image_path = ""
        self._entry_shows_display_path = False
        self._suspend_path_trace = False
        self.current_images = {}
        self.current_result = None
        self._saved_debug_paths = {}
        super().__init__(master, app, FIELD_SPECS, self.params)
        self.image_path_var = tk.StringVar(value="")
        self.image_path_var.trace_add("write", self._on_image_path_changed)
        self.roi_id_var = tk.StringVar()
        self.status_var = tk.StringVar(value="")
        self.debug_image_var = tk.StringVar()
        self._display_to_image_key = {}
        self._build_display_header()
        ttk.Label(self.toolbar, text="Anh khay").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.image_path_var, width=36).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.toolbar, text="Chon anh", command=self.choose_image).pack(side="left", padx=3)
        ttk.Label(self.toolbar, text="ROI ID").pack(side="left", padx=(8, 2))
        self.roi_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_id_var, state="readonly", width=8)
        self.roi_combo.pack(side="left")
        self.roi_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_selected_roi())
        self.run_button = ttk.Button(self.toolbar, text="Run", command=self.run_step)
        self.run_button.pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save ROI", command=self.save_selected_roi).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save Preset", command=self.save_preset).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load Preset", command=self.load_preset_file).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save As...", command=self.save_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load As...", command=self.load_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Reset", command=self.reset_params).pack(side="left", padx=3)
        ttk.Label(self.toolbar, textvariable=self.status_var).pack(side="left", padx=(8, 0))
        self.log_label.pack_forget()
        self.log_panel.pack_forget()
        self._build_result_table()
        self._configure_table_columns()

    def _build_display_header(self):
        self.debug_selector.pack_forget()
        self.display_header = ttk.Frame(self.left_panel)
        self.display_header.pack(fill="x", pady=(0, 6), before=self.image_viewer)
        ttk.Label(self.display_header, text="Anh debug").pack(side="left", padx=(0, 6))
        self.debug_combo = ttk.Combobox(
            self.display_header,
            textvariable=self.debug_image_var,
            state="readonly",
            width=28,
        )
        self.debug_combo.pack(side="left", fill="x", expand=True)
        self.debug_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_select_display_image())

    def _build_result_table(self):
        self.table_section = ttk.Frame(self.left_panel)
        self.table_section.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(self.table_section, text="Bang ket qua ROI refine").pack(anchor="w", pady=(0, 4))
        self.table = ResultTable(self.table_section, ["ID", "center_x", "center_y", "radius", "score", "support_pct"], height=6)
        self.table.pack(fill="x")

    def _configure_table_columns(self):
        width_map = {
            "ID": 70,
            "center_x": 110,
            "center_y": 110,
            "radius": 90,
            "score": 85,
            "support_pct": 100,
        }
        for name, width in width_map.items():
            self.table.tree.column(name, width=width, anchor="center")

    def _display_name_for_key(self, key):
        return DISPLAY_NAME_BY_KEY.get(key, str(key).replace("_", " ").title())

    def _set_debug_options(self, image_keys):
        display_names = [self._display_name_for_key(key) for key in image_keys]
        self._display_to_image_key = {
            self._display_name_for_key(key): key for key in image_keys
        }
        self.debug_combo["values"] = display_names
        current = self.debug_image_var.get()
        if current not in display_names:
            self.debug_image_var.set(display_names[0] if display_names else "")
        return self.debug_image_var.get()

    def _on_select_display_image(self):
        display_name = self.debug_image_var.get()
        image_key = self._display_to_image_key.get(display_name)
        if image_key in self.current_images:
            self.image_viewer.set_image(self.current_images[image_key])
        self._show_current_image_path(image_key)

    def set_result(self, result):
        self.current_result = result
        self._saved_debug_paths = {}
        self.current_images = result.get("images", {}).copy()
        image_keys = list(self.current_images.keys())
        selected = self._set_debug_options(image_keys)
        image_key = self._display_to_image_key.get(selected)
        if image_key in self.current_images:
            self.image_viewer.set_image(self.current_images[image_key])
        elif image_keys:
            image_key = image_keys[0]
            self.image_viewer.set_image(self.current_images[image_key])
        self._show_current_image_path(image_key)
        self.log_panel.set_lines(result.get("logs", []))

    def choose_image(self):
        path = filedialog.askopenfilename(title="Chon anh khay", initialdir=str(INPUT_DIR))
        if path:
            self._source_image_path = path
            self._set_image_path_entry(path, display_only=False)
            if self.auto_update_var.get():
                self.run_step()

    def _store_shared_params(self):
        self.app.shared["roi_params"] = self.params

    def save_preset(self):
        self.params = self.parameter_panel.get_data()
        save_preset(ROI_PRESET_PATH, self.params)
        self._store_shared_params()
        messagebox.showinfo("Preset", "Da luu roi_preset.json")

    def load_preset_file(self):
        self.params = load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS)
        self.parameter_panel.set_data(self.params)
        self._store_shared_params()

    def save_preset_as(self):
        target_path = ask_save_preset_path(ROI_PRESET_PATH, "Luu ROI preset thanh file rieng")
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
        target_path = ask_load_preset_path(ROI_PRESET_PATH, "Nap ROI preset tu file rieng")
        if not target_path:
            return
        self.params = load_preset(target_path, DEFAULT_ROI_PARAMS)
        self.parameter_panel.set_data(self.params)
        self._store_shared_params()
        messagebox.showinfo(
            "Preset",
            "Da nap preset test tu:\n{}\n\nPreset goc trong thu muc presets khong bi ghi de.".format(target_path),
        )

    def reset_params(self):
        self.parameter_panel.set_data(DEFAULT_ROI_PARAMS)
        self.params = self.parameter_panel.get_data()
        self._store_shared_params()

    def _crop_params_only(self, params):
        return {
            "half_size_scale": float(params.get("half_size_scale", 1.30)),
            "output_size": int(round(float(params.get("output_size", 0)))),
        }

    def _build_crop_signature(self, image_path, circles, params):
        return {
            "image_path": image_path,
            "crop_params": self._crop_params_only(params),
            "circles": [
                (
                    int(item.get("id", 0)),
                    round(float(item.get("x", item.get("center_x", 0))), 3),
                    round(float(item.get("y", item.get("center_y", 0))), 3),
                    round(float(item.get("r", item.get("radius", 0))), 3),
                )
                for item in circles
            ],
        }

    def _find_selected_roi_item(self, result, roi_id=None):
        roi_id = (roi_id or self.roi_id_var.get()).strip()
        if not result or not result.get("success") or "rois" not in result.get("data", {}):
            return None
        selected = find_roi_item(result["data"]["rois"], roi_id)
        if selected is not None:
            return selected
        return result["data"]["rois"][0] if result["data"]["rois"] else None

    def _apply_selected_images(self, result):
        selected = self._find_selected_roi_item(result)
        if selected is None:
            return
        effective_item = self.app.shared.get("roi_effective_items", {}).get(selected["id"])
        result["images"]["selected_roi"] = selected["roi"]
        debug_images = effective_item.get("debug", {}) if effective_item is not None else {}
        result["images"]["selected_roi_detected"] = debug_images.get("selected_roi_detected")
        result["images"]["selected_roi_preprocessed"] = debug_images.get("selected_roi_preprocessed")
        result["images"]["selected_roi_edges"] = debug_images.get("selected_roi_edges")
        result["images"]["selected_roi_masked_edges"] = debug_images.get("selected_roi_masked_edges")

    def refresh_selected_roi(self):
        result = self.app.shared.get("roi_result")
        if not result or not result["success"]:
            return
        self._apply_selected_images(result)
        self._apply_summary(result)
        self.set_result(result)

    def save_selected_roi(self):
        result = self.current_result or self.app.shared.get("roi_result")
        if not result or not result["success"]:
            messagebox.showwarning("Chua co anh", "Hay run buoc ROI truoc.")
            return
        image_key = self._display_to_image_key.get(self.debug_image_var.get())
        if not image_key:
            messagebox.showwarning("Chua chon anh", "Hay chon anh debug can luu.")
            return
        image = self.current_images.get(image_key)
        if image is None:
            messagebox.showwarning("Khong co du lieu", "Anh debug hien tai chua co du lieu de luu.")
            return
        from src.io_utils import write_image

        current_path = self._resolve_display_path(image_key)
        initial_dir = str(Path(current_path).parent) if current_path else str(ROI_DIR)
        target_path = filedialog.asksaveasfilename(
            title="Luu anh debug hien tai",
            initialdir=initial_dir,
            initialfile=self._suggest_debug_filename(image_key),
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("Bitmap image", "*.bmp"),
                ("TIFF image", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not target_path:
            return
        path = write_image(target_path, image)
        self._saved_debug_paths[image_key] = str(path)
        messagebox.showinfo("Save ROI", "Da luu anh debug tai:\n{}".format(path))
        self._show_current_image_path(image_key)

    def run_step(self):
        image_path = self._get_request_image_path()
        if not image_path:
            messagebox.showwarning("Thieu anh", "Hay chon anh khay truoc.")
            return
        self.params = self.parameter_panel.get_data()
        shared_hough = None
        if self.app.shared.get("image_path") == image_path:
            shared_hough = self.app.shared.get("hough_result")
        hough_params = self.app.shared.get("hough_params") or load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS)
        request = {
            "image_path": image_path,
            "roi_params": self.params,
            "hough_result": shared_hough,
            "hough_params": hough_params,
            "selected_id": self.roi_id_var.get().strip(),
            "existing_crop_result": self.app.shared.get("roi_crop_result"),
            "existing_crop_signature": self.app.shared.get("roi_crop_signature"),
        }
        self._request_run(request)

    def _request_run(self, request):
        if self._busy:
            self._pending = request
            self.status_var.set("Dang chay... se cap nhat lai")
            return
        self._start_worker(request)

    def _start_worker(self, request):
        self._busy = True
        self.run_button.configure(state="disabled")
        self.status_var.set("Dang xu ly ROI o thread nen")
        self.log_panel.set_lines(
            [
                "Dang cat ROI neu can, sau do Hough tinh tren ROI ID dang chon...",
                "GUI van responsive trong luc tinh toan.",
            ]
        )

        def worker():
            try:
                hough_result = request["hough_result"]
                if not hough_result:
                    hough_result = run_step_hough(request["image_path"], request["hough_params"])

                crop_result = None
                refine_result = None
                crop_signature = None
                selected_base = None

                if hough_result["success"]:
                    crop_signature = self._build_crop_signature(
                        request["image_path"],
                        hough_result["data"]["circles_filtered"],
                        request["roi_params"],
                    )
                    reuse_crop = (
                        request["existing_crop_result"] is not None
                        and request["existing_crop_result"].get("success")
                        and request["existing_crop_signature"] == crop_signature
                    )
                    if reuse_crop:
                        crop_result = request["existing_crop_result"]
                    else:
                        crop_result = run_step_roi_crop(
                            hough_result["images"]["original"],
                            hough_result["data"]["circles_filtered"],
                            request["roi_params"],
                            save_all=True,
                        )
                    if crop_result["success"]:
                        selected_base = self._find_selected_roi_item(crop_result, request["selected_id"])
                        if selected_base is not None:
                            refine_result = run_step_roi_refine(selected_base, request["roi_params"])
                    result = self._compose_result(crop_result, refine_result, selected_base)
                else:
                    result = hough_result
                error = None
            except Exception as exc:
                result, hough_result, crop_result, refine_result, crop_signature, error = None, None, None, None, None, exc
            self._event_queue.put((result, hough_result, crop_result, refine_result, crop_signature, error, request))

        threading.Thread(target=worker, daemon=True).start()
        if self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_worker)

    def _compose_result(self, crop_result, refine_result, selected_base):
        if crop_result is None:
            return {"success": False, "data": {}, "images": {}, "logs": ["Khong tao duoc ROI."]}
        if not crop_result.get("success"):
            return crop_result
        rois = crop_result["data"]["rois"]
        refined_rois = {}
        logs = list(crop_result.get("logs", []))
        if refine_result is not None:
            if refine_result.get("success"):
                refined_item = refine_result["data"]["roi_item"]
                refined_rois[refined_item["id"]] = refined_item
                logs.extend(["ROI ID{:02d}: {}".format(refined_item["id"], msg) for msg in refine_result.get("logs", [])])
            else:
                logs.extend(refine_result.get("logs", []))
        selected_base = selected_base or (rois[0] if rois else None)
        selected_id = selected_base["id"] if selected_base is not None else None
        effective_rois = [refined_rois.get(item["id"], item) for item in rois]
        images = {
            "overview": crop_result["images"].get("overview"),
            "selected_roi": selected_base["roi"] if selected_base is not None else None,
            "selected_roi_detected": None,
            "selected_roi_preprocessed": None,
            "selected_roi_edges": None,
            "selected_roi_masked_edges": None,
        }
        if selected_id in refined_rois and refine_result is not None and refine_result.get("success"):
            images.update(refine_result["images"])
        return {
            "success": True,
            "data": {
                "rois": rois,
                "effective_rois": effective_rois,
                "refined_rois": refined_rois,
                "selected_id": selected_id,
                "saved_paths": crop_result["data"].get("saved_paths", []),
            },
            "images": images,
            "logs": logs,
        }

    def _poll_worker(self):
        self._poll_id = None
        try:
            while True:
                payload = self._event_queue.get_nowait()
                self._on_worker_done(*payload)
        except queue.Empty:
            pass
        if self._busy and self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_worker)

    def _on_worker_done(self, result, hough_result, crop_result, _refine_result, crop_signature, error, request):
        self._busy = False
        self.run_button.configure(state="normal")
        self.status_var.set("")
        if error is not None:
            result = {"success": False, "data": {}, "images": {}, "logs": [str(error)]}

        if result["success"] and "rois" in result.get("data", {}):
            ids = [str(item["id"]) for item in result["data"]["rois"]]
            selected_id = str(result["data"].get("selected_id", ids[0] if ids else ""))
            self.roi_combo["values"] = ids
            if selected_id in ids:
                self.roi_id_var.set(selected_id)
            elif ids:
                self.roi_id_var.set(ids[0])
            self.app.shared["roi_effective_items"] = dict(result.get("data", {}).get("refined_rois", {}))
            self._apply_selected_images(result)
            self._apply_summary(result)
            self.app.shared["roi_result"] = result
            if crop_result is not None and crop_result.get("success"):
                self.app.shared["roi_crop_result"] = crop_result
                self.app.shared["roi_crop_signature"] = crop_signature
            self.app.shared["roi_params"] = request["roi_params"]
            self.app.shared["image_path"] = request["image_path"]
            self._source_image_path = request["image_path"]
            if hough_result and hough_result.get("success"):
                self.app.shared["hough_result"] = hough_result
                self.app.shared["hough_params"] = request["hough_params"]
        else:
            self.table.set_rows([])

        self.set_result(result)

        pending = self._pending
        self._pending = None
        if pending is not None:
            self._start_worker(pending)

    def _apply_summary(self, result):
        selected = self._find_selected_roi_item(result)
        refined_items = []
        if selected is not None:
            effective_item = self.app.shared.get("roi_effective_items", {}).get(selected["id"])
            if effective_item is None:
                effective_item = result.get("data", {}).get("refined_rois", {}).get(selected["id"])
            if effective_item is not None:
                refined_items = [effective_item]
        rows = [
            (
                item["id"],
                int(round(float(item.get("center_x", item["circle"]["x"])))),
                int(round(float(item.get("center_y", item["circle"]["y"])))),
                int(round(float(item.get("radius_full", item["circle"]["r"])))),
                round(float(item.get("score", item["circle"].get("score", 0.0))), 4),
                round(float(item.get("support_pct", item["circle"].get("support_pct", 0.0))), 1),
            )
            for item in refined_items
        ]
        self.table.set_rows(rows)

    def _on_image_path_changed(self, *_args):
        if self._suspend_path_trace:
            return
        self._entry_shows_display_path = False
        self._source_image_path = self.image_path_var.get().strip()

    def _set_image_path_entry(self, path, display_only=False):
        self._suspend_path_trace = True
        try:
            self.image_path_var.set(path or "")
        finally:
            self._suspend_path_trace = False
        self._entry_shows_display_path = bool(display_only and path)

    def _get_request_image_path(self):
        entry_path = self.image_path_var.get().strip()
        if not self._entry_shows_display_path and entry_path:
            self._source_image_path = entry_path
            return entry_path
        if self._source_image_path:
            return self._source_image_path
        shared_path = self.app.shared.get("image_path", "")
        if shared_path:
            self._source_image_path = shared_path
        return shared_path

    def _resolve_selected_roi_path(self):
        selected = self._find_selected_roi_item(self.current_result)
        if selected is None:
            return ""
        saved_path = selected.get("saved_path")
        if saved_path:
            return str(saved_path)
        roi_id = int(selected.get("id", 0))
        if roi_id <= 0:
            return ""
        return str(ROI_DIR / "roi_stator_{:02d}.png".format(roi_id))

    def _resolve_display_path(self, image_key):
        saved_debug_path = self._saved_debug_paths.get(image_key)
        if saved_debug_path:
            return saved_debug_path
        if image_key and image_key.startswith("selected_roi"):
            roi_path = self._resolve_selected_roi_path()
            if roi_path:
                return roi_path
        return self._source_image_path or self.app.shared.get("image_path", "")

    def _show_current_image_path(self, image_key):
        path = self._resolve_display_path(image_key)
        self._set_image_path_entry(path, display_only=bool(path))

    def _suggest_debug_filename(self, image_key):
        source_path = self._source_image_path or self.app.shared.get("image_path", "")
        source_stem = Path(source_path).stem if source_path else "roi_debug"
        selected = self._find_selected_roi_item(self.current_result)
        if image_key == "overview":
            return "{}_roi_overview.png".format(source_stem)
        if image_key and image_key.startswith("selected_roi"):
            roi_suffix = "roi"
            if selected is not None:
                roi_suffix = "roi_{:02d}".format(int(selected.get("id", 0)))
            return "{}_{}_{}.png".format(source_stem, roi_suffix, image_key)
        return "{}_{}.png".format(source_stem, image_key or "debug")
