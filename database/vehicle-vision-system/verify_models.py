#!/usr/bin/env python3
"""Verify the model artifacts required by a clean checkout."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.utils.model_integrity import ModelIntegrityError, verify_model_from_manifest


MODEL_CHECKS = (
    (
        ROOT.parent / "ctpgr-pytorch-master" / "checkpoints" / "lstm_yolo11s.pt",
        ROOT.parent / "ctpgr-pytorch-master" / "checkpoints" / "model_manifest.json",
    ),
    (
        ROOT / "backend" / "app" / "models" / "fh02.pth",
        ROOT / "backend" / "app" / "models" / "model_manifest.json",
    ),
)


def main() -> int:
    try:
        for model_path, manifest_path in MODEL_CHECKS:
            metadata = verify_model_from_manifest(model_path, manifest_path)
            print(
                f"OK {model_path.name}: {metadata['size_bytes']} bytes, "
                f"sha256={metadata['sha256']}"
            )
    except ModelIntegrityError as exc:
        print(f"模型校验失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
