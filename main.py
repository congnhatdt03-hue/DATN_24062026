"""Simple CLI entrypoint for the full pipeline."""

from src.config import (
    DEFAULT_HOUGH_PARAMS,
    DEFAULT_ROI_PARAMS,
    HOUGH_PRESET_PATH,
    INPUT_DIR,
    ROI_PRESET_PATH,
    TEMPLATE_DATA_PATH,
)
from src.io_utils import ensure_project_dirs, get_first_image
from src.pipeline_runner import run_full_pipeline
from src.preset_store import ensure_default_presets, load_preset, load_radial_signature_preset
from src.template_builder import load_template_data


def main():
    """Run the full pipeline on the first image in data/input."""
    ensure_project_dirs()
    ensure_default_presets()
    first_image = get_first_image(INPUT_DIR)
    if first_image is None:
        print("Chua co anh trong data/input/.")
        print("Hay dat anh khay vao data/input/ roi chay lai: python main.py")
        return
    try:
        template_data = load_template_data(TEMPLATE_DATA_PATH)
    except Exception:
        print("Chua co template_data.json.")
        print("Hay tao template trong GUI truoc, sau do chay lai: python main.py")
        return

    hough_params = load_preset(HOUGH_PRESET_PATH, DEFAULT_HOUGH_PARAMS)
    roi_params = load_preset(ROI_PRESET_PATH, DEFAULT_ROI_PARAMS)
    preset_bundle = load_radial_signature_preset()
    tab_edge_params = preset_bundle["tab_edge_params"]
    radial_params = preset_bundle["radial_params"]

    result = run_full_pipeline(str(first_image), hough_params, roi_params, tab_edge_params, radial_params, template_data)
    if not result["success"]:
        print("Pipeline chua chay thanh cong:")
        for line in result["logs"]:
            print("-", line)
        return

    print("Da xong full pipeline.")
    print("Final result:", result["data"]["final_result_path"])
    print("Results CSV :", result["data"]["results_csv_path"])


if __name__ == "__main__":
    main()
