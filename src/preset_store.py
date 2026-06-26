"""Preset loading and saving utilities."""

import copy
import json
from pathlib import Path

from .config import (
    DEFAULT_HOUGH_PARAMS,
    DEFAULT_RADIAL_PARAMS,
    DEFAULT_ROI_PARAMS,
    DEFAULT_TAB_EDGE_PARAMS,
    HOUGH_PRESET_PATH,
    RADIAL_SIGNATURE_PRESET_PATH,
    RADIAL_PRESET_PATH,
    ROI_PRESET_PATH,
    TAB_EDGE_PRESET_PATH,
)
from .io_utils import ensure_dir


def merge_dict(default, override):
    """Recursively merge override into a copy of default."""
    merged = copy.deepcopy(default)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_preset(path, default):
    """Load a JSON preset and merge missing keys with defaults."""
    path_obj = Path(path)
    if not path_obj.is_file():
        return copy.deepcopy(default)
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "config" in payload and isinstance(payload["config"], dict):
        payload = payload["config"]
    return merge_dict(default, payload)


def save_preset(path, data):
    """Save a preset as readable JSON."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path_obj


def default_radial_signature_preset():
    """Return the default combined preset for tab-edges plus radial."""
    return {
        "tab_edge_params": copy.deepcopy(DEFAULT_TAB_EDGE_PARAMS),
        "radial_params": copy.deepcopy(DEFAULT_RADIAL_PARAMS),
    }


def load_radial_signature_preset(path=RADIAL_SIGNATURE_PRESET_PATH):
    """Load the combined tab-edge + radial preset with legacy fallback."""
    path_obj = Path(path)
    if path_obj.is_file():
        payload = load_preset(path_obj, default_radial_signature_preset())
        if isinstance(payload, dict):
            tab_edge_payload = payload.get("tab_edge_params", payload.get("tab_edge", {}))
            radial_payload = payload.get("radial_params", payload.get("radial", {}))
            return {
                "tab_edge_params": merge_dict(DEFAULT_TAB_EDGE_PARAMS, tab_edge_payload),
                "radial_params": merge_dict(DEFAULT_RADIAL_PARAMS, radial_payload),
            }
    return {
        "tab_edge_params": load_preset(TAB_EDGE_PRESET_PATH, DEFAULT_TAB_EDGE_PARAMS),
        "radial_params": load_preset(RADIAL_PRESET_PATH, DEFAULT_RADIAL_PARAMS),
    }


def save_radial_signature_preset(path, tab_edge_params, radial_params):
    """Save one JSON bundle for both tab-edge and radial settings."""
    payload = {
        "tab_edge_params": copy.deepcopy(tab_edge_params),
        "radial_params": copy.deepcopy(radial_params),
    }
    return save_preset(path, payload)


def ensure_default_presets():
    """Create default preset files when missing."""
    preset_map = {
        HOUGH_PRESET_PATH: DEFAULT_HOUGH_PARAMS,
        ROI_PRESET_PATH: DEFAULT_ROI_PARAMS,
        TAB_EDGE_PRESET_PATH: DEFAULT_TAB_EDGE_PARAMS,
        RADIAL_PRESET_PATH: DEFAULT_RADIAL_PARAMS,
    }
    for path, default in preset_map.items():
        if not Path(path).is_file():
            save_preset(path, default)
    if not Path(RADIAL_SIGNATURE_PRESET_PATH).is_file():
        legacy_bundle = load_radial_signature_preset()
        save_radial_signature_preset(
            RADIAL_SIGNATURE_PRESET_PATH,
            legacy_bundle["tab_edge_params"],
            legacy_bundle["radial_params"],
        )
