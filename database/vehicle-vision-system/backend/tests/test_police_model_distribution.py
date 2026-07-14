import json
import sys
from pathlib import Path

import pytest
import torch

from app.utils.model_integrity import ModelIntegrityError, verify_model_from_manifest


DATABASE_ROOT = Path(__file__).resolve().parents[3]
CTPGR_ROOT = DATABASE_ROOT / "ctpgr-pytorch-master"
CHECKPOINT_DIR = CTPGR_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "lstm_yolo11s.pt"
MANIFEST_PATH = CHECKPOINT_DIR / "model_manifest.json"


def test_distributed_police_model_matches_manifest():
    metadata = verify_model_from_manifest(MODEL_PATH, MANIFEST_PATH)

    assert metadata is not None
    assert metadata["architecture"] == "GestureRecognitionModel"
    assert metadata["num_classes"] == 9


def test_distributed_police_model_loads_into_classifier():
    root_str = str(CTPGR_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from models.gesture_recognition_model import GestureRecognitionModel

    model = GestureRecognitionModel(1).cpu()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        feature_count = model.rnn.input_size
        _, _, _, logits = model(
            torch.zeros((1, 1, feature_count), dtype=torch.float32),
            torch.zeros((1, 1, model.num_hidden), dtype=torch.float32),
            torch.zeros((1, 1, model.num_hidden), dtype=torch.float32),
        )
    assert logits.shape == (1, 9)


def test_lfs_pointer_has_actionable_error(tmp_path: Path):
    model_path = tmp_path / "lstm_yolo11s.pt"
    model_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:2046e96b1e6ef986f5bad4a823ceba3d890131ca2be83ec903c088042c7d30c8\n"
        "size 61902\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(json.dumps({"models": {}}), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="git lfs pull"):
        verify_model_from_manifest(model_path, manifest_path)
