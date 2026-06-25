"""Shared file dialogs for loading and saving JSON files."""

from pathlib import Path
from tkinter import filedialog


JSON_FILE_TYPES = [
    ("JSON files", "*.json"),
    ("All files", "*.*"),
]

IMAGE_FILE_TYPES = [
    ("PNG files", "*.png"),
    ("JPEG files", "*.jpg;*.jpeg"),
    ("Bitmap files", "*.bmp"),
    ("TIFF files", "*.tif;*.tiff"),
    ("All files", "*.*"),
]


def ask_save_json_path(default_path, title):
    """Open a Save As dialog seeded from the given JSON location."""
    default_path = Path(default_path)
    return filedialog.asksaveasfilename(
        title=title,
        initialdir=str(default_path.parent),
        initialfile=default_path.name,
        defaultextension=".json",
        filetypes=JSON_FILE_TYPES,
    )


def ask_load_json_path(default_path, title):
    """Open a Load dialog seeded from the given JSON location."""
    default_path = Path(default_path)
    return filedialog.askopenfilename(
        title=title,
        initialdir=str(default_path.parent),
        initialfile=default_path.name,
        filetypes=JSON_FILE_TYPES,
    )


def ask_save_image_path(initialdir, initialfile, title):
    """Open a Save As dialog for an image file."""
    return filedialog.asksaveasfilename(
        title=title,
        initialdir=str(Path(initialdir)),
        initialfile=initialfile,
        defaultextension=".png",
        filetypes=IMAGE_FILE_TYPES,
    )


def ask_save_preset_path(default_path, title):
    """Open a Save As dialog seeded from the default preset location."""
    return ask_save_json_path(default_path, title)


def ask_load_preset_path(default_path, title):
    """Open a Load As dialog seeded from the default preset location."""
    return ask_load_json_path(default_path, title)
