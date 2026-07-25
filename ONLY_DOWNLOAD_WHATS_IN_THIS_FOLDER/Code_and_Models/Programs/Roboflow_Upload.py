from pathlib import Path
import os
import sys


API_KEY = os.getenv("ROBOFLOW_API_KEY", "MVfYFHA3tspeHOGaZ81u")
WORKSPACE_SLUG = os.getenv("ROBOFLOW_WORKSPACE", "final-project-otter")
PROJECT_ID = os.getenv("ROBOFLOW_PROJECT_ID", "billiards-zlynh")
MODEL_TYPE = os.getenv("ROBOFLOW_MODEL_TYPE", "yolov11n")
MODEL_NAME = os.getenv("ROBOFLOW_MODEL_NAME", "Billiards-Detector-YOLO11n")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "Trained_Models" / "YOLO11n"
WEIGHTS_FILE = MODEL_PATH / "weights" / "best.pt"


def main() -> int:
    if not WEIGHTS_FILE.is_file():
        print(f"Model file not found: {WEIGHTS_FILE}")
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow package is not installed.")
        print("Install it with: pip install roboflow")
        return 1

    print("Uploading YOLO11n model to Roboflow...")
    print(f"Workspace : {WORKSPACE_SLUG}")
    print(f"Project   : {PROJECT_ID}")
    print(f"Model     : {WEIGHTS_FILE}")

    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace(WORKSPACE_SLUG)

    workspace.deploy_model(
        model_type=MODEL_TYPE,
        model_path=str(MODEL_PATH),
        project_ids=[PROJECT_ID],
        model_name=MODEL_NAME,
        filename="weights/best.pt",
    )

    print("Model successfully deployed to Roboflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
