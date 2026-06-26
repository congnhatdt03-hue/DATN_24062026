"""Main Tkinter application for detect_stator."""

import tkinter as tk
from tkinter import ttk

from src.io_utils import ensure_project_dirs
from src.preset_store import ensure_default_presets

from .full_pipeline_panel import FullPipelinePanel
from .hough_step_panel import HoughStepPanel
from .matching_step_panel import MatchingStepPanel
from .radial_signature_panel import RadialSignaturePanel
from .roi_step_panel import RoiStepPanel
from .template_step_panel import TemplateStepPanel


class StatorVisionApp:
    """Tkinter app that hosts all 7 pipeline step panels."""

    def __init__(self, root):
        self.root = root
        self.shared = {}
        ensure_project_dirs()
        ensure_default_presets()

        root.title("detect_stator - Step Debug GUI")
        root.geometry("1680x980")
        root.minsize(1320, 840)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.hough_panel = HoughStepPanel(notebook, self)
        self.roi_panel = RoiStepPanel(notebook, self)
        self.radial_signature_panel = RadialSignaturePanel(notebook, self)
        self.tab_edge_panel = self.radial_signature_panel
        self.radial_panel = self.radial_signature_panel
        self.template_panel = TemplateStepPanel(notebook, self)
        self.matching_panel = MatchingStepPanel(notebook, self)
        self.full_pipeline_panel = FullPipelinePanel(notebook, self)

        notebook.add(self.hough_panel, text="1. Hough")
        notebook.add(self.roi_panel, text="2. ROI")
        notebook.add(self.radial_signature_panel, text="3-4. Radial Signature")
        notebook.add(self.template_panel, text="5. Template 0 degree")
        notebook.add(self.matching_panel, text="6. Match MSE")
        notebook.add(self.full_pipeline_panel, text="7. Full Pipeline")
