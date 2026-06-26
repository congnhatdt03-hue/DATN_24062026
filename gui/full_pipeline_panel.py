"""GUI panel for step 7: full tray pipeline.

Nhap anh toan khay + template mau, chuong trinh tu chay luong tab 1 -> tab 6 (coarse
Hough -> ROI -> Hough refine -> tab edge -> radial -> MSE) o thread nen va tra ve bang
ket qua giong tab 6. Tham so lay tu cac preset da luu o tab truoc.
"""

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import (
    DEFAULT_HOUGH_PARAMS,
    DEFAULT_ROI_PARAMS,
    HOUGH_PRESET_PATH,
    INPUT_DIR,
    ROI_PRESET_PATH,
    TEMPLATE_DATA_PATH,
)
from src.pipeline_runner import run_full_pipeline
from src.preset_store import load_preset, load_radial_signature_preset
from src.template_builder import load_template_data
from src.visualization import normalize_display_angle_deg

from .common_widgets import ResultTable, StepPanelBase


POLL_MS = 60


class FullPipelinePanel(StepPanelBase):
    """Panel for running the complete tray pipeline in one shot."""

    def __init__(self, master, app):
        self._busy = False
        self._pending = None
        self._event_queue = queue.Queue()
        self._poll_id = None
        super().__init__(master, app, [], {})
        self.template_data = None
        self.image_path_var = tk.StringVar(value="")
        self.roi_id_var = tk.StringVar()
        self.status_var = tk.StringVar(value="")

        ttk.Label(self.toolbar, text="Anh khay").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.image_path_var, width=30).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.toolbar, text="Chon anh", command=self.choose_image).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load Template", command=self.load_template).pack(side="left", padx=3)
        ttk.Label(self.toolbar, text="ROI ID").pack(side="left", padx=(8, 2))
        self.roi_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_id_var, state="readonly", width=8)
        self.roi_combo.pack(side="left")
        self.roi_combo.bind("<<ComboboxSelected>>", lambda _event: self.show_selected_debug())
        self.run_button = ttk.Button(self.toolbar, text="Run Full Pipeline", command=self.run_step)
        self.run_button.pack(side="left", padx=3)
        ttk.Label(self.toolbar, textvariable=self.status_var).pack(side="left", padx=(8, 0))

        self._build_right_table()

    def _build_right_table(self):
        """Bang ket qua o cot phai (thay cho Log), giong tab 6."""
        self.parameter_panel.pack_forget()
        self.log_label.pack_forget()
        self.log_panel.pack_forget()
        ttk.Label(self.right_panel, text="Bang ket qua toan khay").pack(anchor="w", pady=(2, 4))
        self.table = ResultTable(
            self.right_panel,
            ["ID", "center_x", "center_y", "radius", "angle_deg", "min_error", "status"],
            height=14,
        )
        self.table.pack(fill="x")
        self.log_label.pack(anchor="w", pady=(8, 4))
        self.log_panel.pack(fill="both", expand=True)

    def choose_image(self):
        path = filedialog.askopenfilename(title="Chon anh khay", initialdir=str(INPUT_DIR))
        if path:
            self.image_path_var.set(path)

    def load_template(self):
        try:
            self.template_data = load_template_data(TEMPLATE_DATA_PATH)
            messagebox.showinfo("Template", "Da nap template_data.json")
        except Exception as exc:
            messagebox.showerror("Template", str(exc))

    def show_selected_debug(self):
        result = self.app.shared.get("full_pipeline_result")
        if not result or not result["success"]:
            return
        roi_id = self.roi_id_var.get().strip()
        roi_map = result["data"].get("roi_debug", {})
        tab_map = result["data"].get("tab_edge_debug", {})
        if roi_id:
            result["images"]["roi_selected"] = roi_map.get("roi_{:02d}".format(int(roi_id)))
            result["images"]["tab_edges_selected"] = tab_map.get("tab_edges_{:02d}".format(int(roi_id)))
        self.set_result(result)

    def run_step(self):
        image_path = self.image_path_var.get().strip() or self.app.shared.get("image_path", "")
        if not image_path:
            messagebox.showwarning("Full Pipeline", "Hay chon anh khay truoc.")
            return
        if self._busy:
            self._pending = True
            self.status_var.set("Dang chay... se cap nhat lai")
            return
        template_data = self.template_data or self.app.shared.get("template_data")
        if template_data is None:
            try:
                template_data = load_template_data(TEMPLATE_DATA_PATH)
            except Exception as exc:
                messagebox.showwarning("Template", "Chua co template. Hay tao o tab 5. ({})".format(exc))
                return
        self.template_data = template_data
        preset_bundle = load_radial_signature_preset()
        request = {
            "image_path": image_path,
            "hough_params": self.app.shared.get("hough_params") or load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS),
            "roi_params": self.app.shared.get("roi_params") or load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS),
            "tab_edge_params": self.app.shared.get("tab_edge_params") or preset_bundle["tab_edge_params"],
            "radial_params": self.app.shared.get("radial_params") or preset_bundle["radial_params"],
            "template_data": template_data,
        }
        self._start_worker(request)

    def _start_worker(self, request):
        self._busy = True
        self.run_button.configure(state="disabled")
        self.status_var.set("Dang chay full pipeline...")
        self.log_panel.set_lines(
            [
                "Dang chay full pipeline o thread nen (GUI van responsive)...",
                "Coarse Hough (fast mode neu bat) -> ROI -> Hough refine -> tab edge -> radial -> MSE.",
            ]
        )

        def worker():
            try:
                result = run_full_pipeline(
                    request["image_path"],
                    request["hough_params"],
                    request["roi_params"],
                    request["tab_edge_params"],
                    request["radial_params"],
                    request["template_data"],
                    save_debug=False,
                )
                error = None
            except Exception as exc:
                result, error = None, exc
            self._event_queue.put((result, error))

        threading.Thread(target=worker, daemon=True).start()
        if self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_worker)

    def _poll_worker(self):
        self._poll_id = None
        try:
            while True:
                result, error = self._event_queue.get_nowait()
                self._on_worker_done(result, error)
        except queue.Empty:
            pass
        if self._busy and self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._poll_worker)

    def _on_worker_done(self, result, error):
        self._busy = False
        self.run_button.configure(state="normal")
        self.status_var.set("")
        if error is not None:
            result = {"success": False, "data": {}, "images": {}, "logs": [str(error)]}
        if result["success"]:
            rows = [
                (
                    row["id"],
                    int(round(row["center_x"])),
                    int(round(row["center_y"])),
                    int(round(row["radius"])),
                    round(normalize_display_angle_deg(row["angle_deg"]), 2),
                    round(row["min_error"], 4),
                    row["status"],
                )
                for row in result["data"]["results"]
            ]
            self.table.set_rows(rows)
            ids = [str(row["id"]) for row in result["data"]["results"]]
            self.roi_combo["values"] = ids
            if ids and self.roi_id_var.get() not in ids:
                self.roi_id_var.set(ids[0])
            self.app.shared["full_pipeline_result"] = result
            self.set_result(result)
        else:
            self.table.set_rows([])
            self.set_result(result)

        if self._pending:
            self._pending = None
            self.run_step()
