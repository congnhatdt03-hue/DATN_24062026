"""GUI panel for step 6: MSE matching by detected stator ID.

Bo cuc: anh ghep mau|test o khung anh chinh (trai-tren), do thi so khop o trai-duoi,
bang ket qua tich luy o cot phai (thay cho Log). Moi lan Run them mot dong test, hang
"mau" luon ghim tren dau de nhin tong quat goc cua tat ca stator.
"""

import io
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DEFAULT_RADIAL_PARAMS,
    DEFAULT_ROI_PARAMS,
    DEFAULT_TAB_EDGE_PARAMS,
    INPUT_DIR,
    RADIAL_PRESET_PATH,
    ROI_PRESET_PATH,
    TAB_EDGE_PRESET_PATH,
    TEMPLATE_DATA_PATH,
)
from src.io_utils import read_image
from src.pipeline_runner import run_step_matching
from src.preset_store import load_preset
from src.roi_extractor import build_roi_item_from_image
from src.template_builder import load_template_bundle
from src.visualization import make_pair_view

from .common_widgets import ImageViewer, ResultTable, StepPanelBase
from .preset_dialogs import ask_load_json_path


def _match_figure(mse_curve, template_norm, current_norm, angle_deg):
    """Render the angle-error curve and the Ref-vs-current signature overlay."""
    curve = np.asarray(mse_curve, dtype=float)
    n = len(curve)
    idx = np.arange(n)
    signed = np.where(idx <= n // 2, idx, idx - n).astype(float)
    order = np.argsort(signed)

    figure, axes = plt.subplots(2, 1, figsize=(7.6, 5.2), dpi=120)
    axes[0].plot(signed[order], curve[order], color="#0ea5e9", linewidth=1.6)
    axes[0].axvline(float(signed[int(np.argmin(curve))]), color="#64748b", linewidth=1.0)
    axes[0].set_title("Do thi sai so binh phuong theo goc (min @ {:.2f} deg)".format(angle_deg))
    axes[0].set_xlabel("Goc lech (deg)")
    axes[0].set_ylabel("Sai so")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(template_norm, color="#0ea5e9", linewidth=1.4, label="Ref (mau)")
    axes[1].plot(current_norm, color="#22c55e", linewidth=1.4, label="Anh hien tai")
    axes[1].set_title("So khop Radial Signature")
    axes[1].set_xlabel("Goc quet 0 -> 360 deg")
    axes[1].set_ylabel("Gia tri signature")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return cv2.imdecode(np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)


class MatchingStepPanel(StepPanelBase):
    """Match one detected stator ID against the saved 0-degree template."""

    def __init__(self, master, app):
        super().__init__(master, app, [], {})
        self.template_data = None
        self.template_roi_image = None
        self._result_rows = {}

        self.roi_path_var = tk.StringVar()
        self.roi_id_var = tk.StringVar()
        ttk.Label(self.toolbar, text="Stator ID").pack(side="left")
        self.roi_combo = ttk.Combobox(self.toolbar, textvariable=self.roi_id_var, state="readonly", width=8)
        self.roi_combo.pack(side="left", padx=(2, 8))
        ttk.Label(self.toolbar, text="hoac file").pack(side="left")
        ttk.Entry(self.toolbar, textvariable=self.roi_path_var, width=24).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(self.toolbar, text="Chon ROI", command=self.choose_roi).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load Template", command=self.load_template).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Load Template File...", command=self.load_template_file).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Run", command=self.run_step).pack(side="left", padx=3)
        ttk.Button(self.toolbar, text="Xoa bang", command=self.clear_table).pack(side="left", padx=3)

        self._build_plot_area()
        self._build_right_table()
        self.refresh_ids()

    def _build_plot_area(self):
        """Khung do thi so khop nam o trai-duoi (cho bang ket qua cu)."""
        host = ttk.Frame(self.left_panel, height=360)
        host.pack(side="bottom", fill="x", pady=(6, 0))
        host.pack_propagate(False)
        ttk.Label(host, text="Do thi so khop").pack(anchor="w")
        self.plot_viewer = ImageViewer(host)
        self.plot_viewer.pack(fill="both", expand=True)

    def _build_right_table(self):
        """Bang ket qua tich luy thay cho Log o cot phai."""
        self.parameter_panel.pack_forget()
        self.log_label.pack_forget()
        self.log_panel.pack_forget()
        ttk.Label(self.right_panel, text="Bang ket qua so khop (mau ghim tren dau)").pack(anchor="w", pady=(2, 4))
        self.table = ResultTable(
            self.right_panel,
            ["doi tuong", "center_x", "center_y", "radius", "angle_deg", "min_error"],
            height=14,
        )
        self.table.pack(fill="x")
        self.log_label.pack(anchor="w", pady=(8, 4))
        self.log_panel.pack(fill="both", expand=True)

    def refresh_ids(self):
        """Populate the stator ID combobox from the ROI step result."""
        roi_result = self.app.shared.get("roi_result")
        ids = [str(item["id"]) for item in roi_result["data"]["rois"]] if roi_result and roi_result["success"] else []
        self.roi_combo["values"] = ids
        if ids and not self.roi_id_var.get():
            self.roi_id_var.set(ids[0])

    def choose_roi(self):
        path = filedialog.askopenfilename(title="Chon ROI test", initialdir=str(INPUT_DIR))
        if path:
            self.roi_path_var.set(path)

    def _apply_loaded_template(self, template_data, template_roi_image):
        self.template_data = template_data
        self.template_roi_image = template_roi_image
        self.app.shared["template_data"] = template_data
        self.app.shared["template_roi_image"] = template_roi_image
        self._rebuild_table()

    def load_template(self):
        try:
            template_data, template_roi_image, _roi_path = load_template_bundle(TEMPLATE_DATA_PATH)
            self._apply_loaded_template(template_data, template_roi_image)
            messagebox.showinfo("Template", "Da nap template_data.json")
        except Exception as exc:
            messagebox.showerror("Template", str(exc))

    def load_template_file(self):
        target_path = ask_load_json_path(TEMPLATE_DATA_PATH, "Nap template tu file ngoai")
        if not target_path:
            return
        try:
            template_data, template_roi_image, _roi_path = load_template_bundle(target_path)
            self._apply_loaded_template(template_data, template_roi_image)
            messagebox.showinfo("Template", "Da nap template ngoai:\n{}".format(target_path))
        except Exception as exc:
            messagebox.showerror("Template", str(exc))

    def clear_table(self):
        """Reset cac dong test da tich luy (giu lai hang mau)."""
        self._result_rows = {}
        self._rebuild_table()

    def _roi_item_by_id(self, roi_id):
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
        """Pick the test ROI: external file first, then selected ID, then first ROI."""
        roi_path = self.roi_path_var.get().strip()
        if roi_path:
            image = read_image(roi_path)
            if image is None:
                raise ValueError("Khong doc duoc ROI test.")
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
        raise ValueError("Hay chon stator ID (chay buoc 2 truoc) hoac chon file ROI test.")

    def _template_image(self):
        return self.template_roi_image if self.template_roi_image is not None else self.app.shared.get("template_roi_image")

    @staticmethod
    def _roi_center(roi_item):
        center_in_roi = roi_item.get("center_in_roi", (0.0, 0.0))
        return (
            float(roi_item.get("center_x", center_in_roi[0])),
            float(roi_item.get("center_y", center_in_roi[1])),
        )

    def _rebuild_table(self):
        """Hang mau ghim tren dau, cac dong test tich luy ben duoi theo ID tang dan."""
        template = self.template_data or {}
        t_center = template.get("center_full") or template.get("center", [0.0, 0.0])
        t_radius = float(template.get("radius_full", template.get("radius", 0.0)))
        t_id = template.get("source_id", "-")
        rows = [("mau ID{}".format(t_id), int(round(t_center[0])), int(round(t_center[1])), int(round(t_radius)), "", "")]
        for entry in sorted(self._result_rows.values(), key=lambda item: item["sort_key"]):
            rows.append(entry["row"])
        self.table.set_rows(rows)

    def _record_result(self, roi_item, match_data):
        test_cx, test_cy = self._roi_center(roi_item)
        roi_id = roi_item.get("id", 0)
        self._result_rows[roi_id] = {
            "sort_key": int(roi_id) if str(roi_id).isdigit() else 0,
            "row": (
                "test ID{}".format(roi_id),
                int(round(test_cx)),
                int(round(test_cy)),
                int(round(float(roi_item.get("radius_full", roi_item.get("radius", 0.0))))),
                round(float(match_data["angle_deg"]), 2),
                round(float(match_data["min_error"]), 4),
            ),
        }
        self._rebuild_table()

    def run_step(self):
        self.refresh_ids()
        if self.template_data is None:
            self.template_data = self.app.shared.get("template_data")
        if self.template_data is None:
            try:
                template_data, template_roi_image, _roi_path = load_template_bundle(TEMPLATE_DATA_PATH)
                self._apply_loaded_template(template_data, template_roi_image)
            except Exception:
                messagebox.showwarning("Template", "Chua co template. Hay tao o tab 5 hoac Load Template.")
                return
        try:
            roi_item = self._resolve_roi_item()
        except Exception as exc:
            messagebox.showwarning("Matching", str(exc))
            return
        tab_params = self.app.shared.get("tab_edge_params") or load_preset(TAB_EDGE_PRESET_PATH, DEFAULT_TAB_EDGE_PARAMS)
        radial_params = self.app.shared.get("radial_params") or load_preset(RADIAL_PRESET_PATH, DEFAULT_RADIAL_PARAMS)
        result = run_step_matching(roi_item, self.template_data, tab_params, radial_params)
        roi_logs = roi_item.get("logs") or []
        if roi_logs:
            result["logs"] = list(roi_logs) + list(result.get("logs", []))
        if not result["success"]:
            self.set_result(result)
            return

        match_data = result["data"]
        template_img = self._template_image()
        pair_view = make_pair_view(
            template_img,
            roi_item["roi"],
            left_label="Mau 0 do",
            right_label="Test ID{}".format(roi_item.get("id", "-")),
        )
        figure = _match_figure(
            match_data["mse_curve"],
            self.template_data.get("signature_norm", []),
            match_data["signature_norm"],
            float(match_data["angle_deg"]),
        )
        display = {
            "ket_qua": pair_view,
            "stator_mau": template_img,
            "stator_test": roi_item["roi"],
        }
        for key, image in result["images"].items():
            if key not in display:
                display[key] = image
        result["images"] = display
        self.set_result(result)
        self.plot_viewer.set_image(figure)
        self._record_result(roi_item, match_data)
