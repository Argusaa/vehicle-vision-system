import sys
from pathlib import Path

import torch

from app.utils.model_integrity import verify_model_from_manifest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_ROOT / "app" / "models"
MODEL_PATH = MODEL_DIR / "fh02.pth"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"


def test_distributed_rpnet_model_matches_manifest():
    metadata = verify_model_from_manifest(MODEL_PATH, MANIFEST_PATH)

    assert metadata is not None
    assert metadata["architecture"] == "CCPD RPNet fh02"
    assert metadata["license"] == "MIT"


def test_distributed_rpnet_model_loads_on_cpu():
    backend_str = str(BACKEND_ROOT)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
    from app.ccpd.inference import CCPDRPNetRecognizer

    recognizer = CCPDRPNetRecognizer(str(MODEL_PATH))
    model = recognizer.model

    assert model.training is False
    assert recognizer._device == torch.device("cpu") or torch.cuda.is_available()
    assert sum(parameter.numel() for parameter in model.parameters()) == 55_073_818
