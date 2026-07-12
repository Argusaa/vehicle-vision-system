import importlib
import sys
from pathlib import Path

import numpy as np

from app.services.lpr_video_service import LprVideoService


ASSET_ROOT = Path(__file__).resolve().parents[2] / "yolo_lprnet_assets"


def test_video_runtime_no_longer_depends_on_removed_data_package():
    sys.path.insert(0, str(ASSET_ROOT))
    try:
        runtime = importlib.import_module("runtime_api")
        demo = importlib.import_module("demo_integrated_lpr")
    finally:
        sys.path.remove(str(ASSET_ROOT))

    assert runtime.CHARS
    assert demo.CHARS == runtime.CHARS


def test_video_service_preserves_detected_plate_color():
    class FakeRuntime:
        @staticmethod
        def process_frame(frame):
            return frame, [{
                "text": "皖AF07000",
                "plate_color": "绿牌",
                "coords": (1, 2, 30, 12),
                "confidence": 0.88,
            }]

    service = LprVideoService()
    service._runtime = FakeRuntime()
    service._error = None
    result = service.recognize_frame(np.zeros((24, 64, 3), dtype=np.uint8))

    assert result["model_available"] is True
    assert result["plates"][0]["plate_color"] == "绿牌"
