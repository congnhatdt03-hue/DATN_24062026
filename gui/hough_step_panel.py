"""GUI panel for step 1: Hough detection."""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.config import DEFAULT_HOUGH_PARAMS, HOUGH_PRESET_PATH, INPUT_DIR
from src.io_utils import get_first_image, list_images
from src.pipeline_runner import run_step_hough
from src.preset_store import load_preset, save_preset

from .common_widgets import ResultTable, StepPanelBase
from .preset_dialogs import ask_load_preset_path, ask_save_preset_path


MODE_SINGLE = "Anh don"
MODE_FOLDER = "Thu muc"


FIELD_SPECS = [
    {"path": "expected_count", "label": "Expected count", "type": "int", "group": "Hough", "min": 1, "max": 30},
    {"path": "fast_mode.enabled", "label": "Enable fast mode", "type": "bool", "group": "Fast Mode"},
    {"path": "fast_mode.max_processing_dim", "label": "Max proc dim", "type": "int", "group": "Fast Mode", "min": 320, "max": 3000},
    {"path": "preprocess.enabled", "label": "Enable preprocess", "type": "bool", "group": "Preprocess"},
    {"path": "preprocess.use_clahe", "label": "Use CLAHE", "type": "bool", "group": "Preprocess"},
    {"path": "preprocess.clahe_clip_limit", "label": "CLAHE clip", "type": "float", "group": "Preprocess", "min": 0.1, "max": 10.0},
    {"path": "preprocess.clahe_tile_grid_size", "label": "CLAHE tile", "type": "int", "group": "Preprocess", "min": 1, "max": 32},
    {"path": "preprocess.use_gaussian", "label": "Use Gaussian", "type": "bool", "group": "Preprocess"},
    {"path": "preprocess.gaussian_kernel", "label": "Gaussian kernel", "type": "int", "group": "Preprocess", "min": 1, "max": 31},
    {"path": "hough.dp", "label": "dp", "type": "float", "group": "Hough", "min": 1.0, "max": 3.0},
    {"path": "hough.param1", "label": "param1", "type": "float", "group": "Hough", "min": 1, "max": 300},
    {"path": "hough.param2", "label": "param2", "type": "float", "group": "Hough", "min": 1, "max": 200},
    {"path": "hough.minDist", "label": "minDist", "type": "float", "group": "Hough", "min": 10, "max": 500},
    {"path": "hough.minRadius", "label": "minRadius", "type": "int", "group": "Hough", "min": 1, "max": 300},
    {"path": "hough.maxRadius", "label": "maxRadius", "type": "int", "group": "Hough", "min": 2, "max": 500},
    {"path": "hough.min_center_dist", "label": "min_center_dist", "type": "float", "group": "Hough", "min": 1, "max": 500},
    {"path": "filter.use_radius_consensus", "label": "Radius consensus", "type": "bool", "group": "Filter"},
    {"path": "filter.radius_consensus_tol", "label": "Consensus tol", "type": "int", "group": "Filter", "min": 1, "max": 50},
    {"path": "filter.force_common_radius", "label": "Force common r", "type": "bool", "group": "Filter"},
]

POLL_MS = 60
DISPLAY_NAME_BY_KEY = {
    "original": "Original",
    "preprocessed": "Preprocessed",
    "hough_all": "Hough All",
    "hough_filtered": "Hough Filtered",
}


class HoughStepPanel(StepPanelBase):
    """Panel for tuning and running HoughCircle."""

    def __init__(self, master, app):
        self.params = load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS)
        self._busy = False
        self._pending = None
        self._event_queue = queue.Queue()
        self._poll_id = None
        super().__init__(master, app, FIELD_SPECS, self.params)
        self.image_path_var = tk.StringVar(value=str(get_first_image(INPUT_DIR) or ""))
        self.status_var = tk.StringVar(value="")
        self.detected_var = tk.StringVar(value="Detected: - / -")
        self.debug_image_var = tk.StringVar()
        self._display_to_image_key = {}
        self.input_mode_var = tk.StringVar(value=MODE_SINGLE)
        self.nav_var = tk.StringVar(value="")
        self.folder_images = []
        self.folder_index = 0
        self.summary_rows = {}
        self._build_hough_display_header()
        ttk.Label(self.toolbar, text="Nguon dau vao").pack(side="left")
        self.mode_combo = ttk.Combobox(
            self.toolbar,
            textvariable=self.input_mode_var,
            state="readonly",
            width=9,
            values=[MODE_SINGLE, MODE_FOLDER],
        )
        self.mode_combo.pack(side="left", padx=(2, 8))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_mode_change())
        ttk.Label(self.toolbar, text="Anh khay").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.image_path_var, width=36).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.toolbar, text="Chon anh", command=self.choose_image).pack(side="left", padx=3)
        self.run_button = ttk.Button(self.toolbar, text="Run", command=self.run_step)
        self.run_button.pack(side="left", padx=3)
        self.prev_button = ttk.Button(self.toolbar, text="< Prev", command=self.prev_image, state="disabled")
        self.prev_button.pack(side="left", padx=3)
        self.next_button = ttk.Button(self.toolbar, text="Next >", command=self.next_image, state="disabled")
        self.next_button.pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save Preset", command=self.save_preset).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load Preset", command=self.load_preset_file).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Save As...", command=self.save_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load As...", command=self.load_preset_as).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Reset", command=self.reset_params).pack(side="left", padx=3)
        ttk.Label(self.toolbar, textvariable=self.status_var).pack(side="left", padx=(8, 0))
        ttk.Label(self.left_panel, textvariable=self.nav_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 2), before=self.image_viewer)
        self._build_hough_result_table()
        self._configure_table_columns()
        self._build_overview_table()

    def _build_hough_display_header(self):
        parent = self.left_panel
        self.debug_selector.pack_forget()
        self.display_header = ttk.Frame(parent)
        self.display_header.pack(fill="x", pady=(0, 6), before=self.image_viewer)
        debug_left = ttk.Frame(self.display_header)
        debug_left.pack(side="left", fill="x", expand=True)
        ttk.Label(debug_left, text="Anh debug").pack(side="left", padx=(0, 6))
        self.debug_combo = ttk.Combobox(
            debug_left,
            textvariable=self.debug_image_var,
            state="readonly",
            width=24,
        )
        self.debug_combo.pack(side="left", fill="x", expand=True)
        self.debug_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_select_display_image())
        self.detected_label = tk.Label(
            self.display_header,
            textvariable=self.detected_var,
            font=("Segoe UI", 16, "bold"),
            fg="#555555",
        )
        self.detected_label.pack(side="right", padx=(12, 0))

    def _build_hough_result_table(self):
        self.table_section = ttk.Frame(self.left_panel)
        self.table_section.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(self.table_section, text="Bang ket qua circle").pack(anchor="w", pady=(0, 4))
        self.table = ResultTable(self.table_section, ["ID", "center_x", "center_y", "radius", "score"], height=6)
        self.table.pack(fill="x")

    def _configure_table_columns(self):
        width_map = {
            "ID": 70,
            "center_x": 120,
            "center_y": 120,
            "radius": 100,
            "score": 100,
        }
        for name, width in width_map.items():
            self.table.tree.column(name, width=width, anchor="center")

    def _build_overview_table(self):
        """Thay panel Log ben phai bang bang ket qua tong quat tung anh."""
        self.log_label.pack_forget()
        self.log_panel.pack_forget()
        self.overview_label = ttk.Label(self.right_panel, text="Ket qua tong quat tung anh")
        self.overview_label.pack(anchor="w", pady=(6, 4))
        self.overview_table = ResultTable(
            self.right_panel,
            ["STT", "Anh", "Stator", "Ban kinh TB", "Score TB"],
            height=12,
        )
        self.overview_table.pack(fill="both", expand=True)
        width_map = {"STT": 45, "Anh": 130, "Stator": 70, "Ban kinh TB": 95, "Score TB": 80}
        for name, width in width_map.items():
            self.overview_table.tree.column(name, width=width, anchor="center")

    def _on_mode_change(self):
        """Doi giua che do anh don / thu muc -> reset trang thai duyet."""
        self.folder_images = []
        self.folder_index = 0
        self.image_path_var.set("")
        self._clear_overview()
        self._reset_summary_ui()
        self._update_nav()

    def _load_folder(self, folder):
        """Nap danh sach anh trong thu muc, chua chay."""
        images = list_images(folder)
        if not images:
            messagebox.showwarning("Thu muc rong", "Khong tim thay anh trong thu muc.")
            return
        self.folder_images = images
        self.folder_index = 0
        self.image_path_var.set(str(folder))
        self._clear_overview()
        self._reset_summary_ui()
        self._update_nav()

    def _run_folder_index(self, index):
        """Chay step Hough cho anh tai vi tri index trong thu muc."""
        if not self.folder_images:
            return
        index = max(0, min(index, len(self.folder_images) - 1))
        self.folder_index = index
        image_path = self.folder_images[index]
        self._update_nav()
        self._request_run(str(image_path), self.params, stt=index + 1, image_name=image_path.name)

    def next_image(self):
        """Xu ly anh tiep theo trong thu muc."""
        if self.input_mode_var.get() != MODE_FOLDER or not self.folder_images:
            return
        if self.folder_index < len(self.folder_images) - 1:
            self.params = self.parameter_panel.get_data()
            self._run_folder_index(self.folder_index + 1)

    def prev_image(self):
        """Quay lai anh truoc do trong thu muc."""
        if self.input_mode_var.get() != MODE_FOLDER or not self.folder_images:
            return
        if self.folder_index > 0:
            self.params = self.parameter_panel.get_data()
            self._run_folder_index(self.folder_index - 1)

    def _update_nav(self):
        """Cap nhat nhan vi tri va trang thai nut Prev/Next."""
        is_folder = self.input_mode_var.get() == MODE_FOLDER and bool(self.folder_images)
        if is_folder:
            total = len(self.folder_images)
            name = self.folder_images[self.folder_index].name
            self.nav_var.set("Anh {}/{}: {}".format(self.folder_index + 1, total, name))
            self.prev_button.configure(state="normal" if self.folder_index > 0 else "disabled")
            self.next_button.configure(state="normal" if self.folder_index < total - 1 else "disabled")
        else:
            self.nav_var.set("")
            self.prev_button.configure(state="disabled")
            self.next_button.configure(state="disabled")

    def _clear_overview(self):
        self.summary_rows = {}
        self.overview_table.set_rows([])

    def _update_overview(self, request, result):
        """Upsert mot dong tong quat cho anh vua chay (key theo STT)."""
        circles = result.get("data", {}).get("circles_filtered", [])
        params = request.get("params", {})
        expected = int(round(float(params.get("expected_count", 12))))
        radii = [float(item.get("r", item.get("radius", 0))) for item in circles]
        scores = [float(item.get("score", 0.0)) for item in circles]
        radius_avg = round(sum(radii) / len(radii), 1) if radii else 0.0
        score_avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        stt = int(request.get("stt", 1))
        name = request.get("image_name", "")
        self.summary_rows[stt] = (stt, name, "{}/{}".format(len(circles), expected), radius_avg, score_avg)
        rows = [self.summary_rows[key] for key in sorted(self.summary_rows)]
        self.overview_table.set_rows(rows)

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

    def set_result(self, result):
        self.current_images = result.get("images", {}).copy()
        image_keys = list(self.current_images.keys())
        selected = self._set_debug_options(image_keys)
        image_key = self._display_to_image_key.get(selected)
        if image_key in self.current_images:
            self.image_viewer.set_image(self.current_images[image_key])
        elif image_keys:
            self.image_viewer.set_image(self.current_images[image_keys[0]])
        self.log_panel.set_lines(result.get("logs", []))

    def choose_image(self):
        if self.input_mode_var.get() == MODE_FOLDER:
            folder = filedialog.askdirectory(title="Chon thu muc anh", initialdir=str(INPUT_DIR))
            if folder:
                self._load_folder(folder)
            return
        path = filedialog.askopenfilename(title="Chon anh khay", initialdir=str(INPUT_DIR))
        if path:
            self.image_path_var.set(path)
            self._reset_summary_ui()
            if self.auto_update_var.get():
                self.run_step()

    def _store_shared_params(self):
        self.app.shared["hough_params"] = self.params

    def save_preset(self):
        self.params = self.parameter_panel.get_data()
        save_preset(HOUGH_PRESET_PATH, self.params)
        self._store_shared_params()
        messagebox.showinfo("Preset", "Da luu hough_preset.json")

    def load_preset_file(self):
        self.params = load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS)
        self.parameter_panel.set_data(self.params)
        self._store_shared_params()

    def save_preset_as(self):
        target_path = ask_save_preset_path(HOUGH_PRESET_PATH, "Luu Hough preset thanh file rieng")
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
        target_path = ask_load_preset_path(HOUGH_PRESET_PATH, "Nap Hough preset tu file rieng")
        if not target_path:
            return
        self.params = load_preset(target_path, DEFAULT_HOUGH_PARAMS)
        self.parameter_panel.set_data(self.params)
        self._store_shared_params()
        messagebox.showinfo(
            "Preset",
            "Da nap preset test tu:\n{}\n\nPreset goc trong thu muc presets khong bi ghi de.".format(target_path),
        )

    def reset_params(self):
        self.parameter_panel.set_data(DEFAULT_HOUGH_PARAMS)
        self.params = self.parameter_panel.get_data()
        self._store_shared_params()

    def run_step(self):
        self.params = self.parameter_panel.get_data()
        if self.input_mode_var.get() == MODE_FOLDER:
            if not self.folder_images:
                messagebox.showwarning("Thieu thu muc", "Hay chon thu muc anh truoc.")
                return
            self._run_folder_index(self.folder_index)
            return
        image_path = self.image_path_var.get().strip()
        if not image_path:
            messagebox.showwarning("Thieu anh", "Hay chon anh khay truoc.")
            return
        self._request_run(image_path, self.params, stt=1, image_name=Path(image_path).name)

    def _request_run(self, image_path, params, stt=1, image_name=""):
        request = {
            "image_path": image_path,
            "params": params,
            "stt": stt,
            "image_name": image_name or Path(image_path).name,
        }
        if self._busy:
            self._pending = request
            self.status_var.set("Dang chay... se cap nhat lai")
            return
        self._start_worker(request)

    def _start_worker(self, request):
        self._busy = True
        self.run_button.configure(state="disabled")
        self.status_var.set("Dang detect o thread nen")
        self.detected_var.set("Dang chay Hough...")
        self.detected_label.configure(fg="#888888")
        self.log_panel.set_lines(
            [
                "Dang chay HoughCircle o thread nen...",
                "GUI van responsive trong luc detect.",
            ]
        )

        def worker():
            try:
                result = run_step_hough(request["image_path"], request["params"])
                error = None
            except Exception as exc:
                result, error = None, exc
            self._event_queue.put((result, error, request))

        threading.Thread(target=worker, daemon=True).start()
        if self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_worker)

    def _poll_worker(self):
        self._poll_id = None
        try:
            while True:
                result, error, request = self._event_queue.get_nowait()
                self._on_worker_done(result, error, request)
        except queue.Empty:
            pass
        if self._busy and self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_worker)

    def _on_worker_done(self, result, error, request):
        self._busy = False
        self.run_button.configure(state="normal")
        self.status_var.set("")
        if error is not None:
            result = {"success": False, "data": {}, "images": {}, "logs": [str(error)]}

        self.set_result(result)
        if result["success"]:
            self._apply_summary(result, request["params"])
            self._update_overview(request, result)
            self.app.shared["image_path"] = request["image_path"]
            self.app.shared["hough_result"] = result
            self.app.shared["hough_params"] = request["params"]
        else:
            self._reset_summary_ui()

        pending = self._pending
        self._pending = None
        if pending is not None:
            self._start_worker(pending)

    def _apply_summary(self, result, params):
        circles = result.get("data", {}).get("circles_filtered", [])
        expected = int(round(float(params.get("expected_count", 12))))
        self.detected_var.set("Detected: {} / {}".format(len(circles), expected))
        self.detected_label.configure(fg="#1a8a1a" if len(circles) == expected else "#c0392b")
        rows = [
            (
                item.get("id", index),
                int(round(float(item.get("x", item.get("center_x", 0))))),
                int(round(float(item.get("y", item.get("center_y", 0))))),
                int(round(float(item.get("r", item.get("radius", 0))))),
                round(float(item.get("score", 0.0)), 4),
            )
            for index, item in enumerate(circles, start=1)
        ]
        self.table.set_rows(rows)

    def _reset_summary_ui(self):
        self.detected_var.set("Detected: - / -")
        self.detected_label.configure(fg="#555555")
        self.table.set_rows([])
