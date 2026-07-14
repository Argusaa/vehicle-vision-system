from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class ModelIntegrityError(RuntimeError):
    """Raised when a distributed model is missing or does not match its manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_from_manifest(
    model_path: str | Path,
    manifest_path: str | Path,
    *,
    require_entry: bool = True,
) -> dict[str, Any] | None:
    """Verify a model's size and SHA-256 using a checked-in manifest entry."""

    model_path = Path(model_path)
    manifest_path = Path(manifest_path)
    if not model_path.is_file():
        raise ModelIntegrityError(
            f"模型文件不存在：{model_path}。请安装 Git LFS 后运行 git lfs pull。"
        )
    with model_path.open("rb") as model_file:
        if model_file.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            raise ModelIntegrityError(
                f"{model_path.name} 仍是 Git LFS 指针，请安装 Git LFS 后运行 git lfs pull。"
            )

    if not manifest_path.is_file():
        if require_entry:
            raise ModelIntegrityError(f"模型清单不存在：{manifest_path}")
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelIntegrityError(f"无法读取模型清单：{manifest_path}") from exc

    metadata = manifest.get("models", {}).get(model_path.name)
    if metadata is None:
        if require_entry:
            raise ModelIntegrityError(f"模型清单中没有 {model_path.name} 的记录")
        return None

    expected_size = metadata.get("size_bytes")
    actual_size = model_path.stat().st_size
    if not isinstance(expected_size, int) or actual_size != expected_size:
        raise ModelIntegrityError(
            f"{model_path.name} 大小校验失败：期望 {expected_size} 字节，实际 {actual_size} 字节。"
            "请运行 git lfs pull 后重试。"
        )

    expected_sha256 = str(metadata.get("sha256", "")).lower()
    actual_sha256 = _sha256(model_path)
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise ModelIntegrityError(
            f"{model_path.name} SHA-256 校验失败：期望 {expected_sha256}，实际 {actual_sha256}。"
            "请删除损坏文件并运行 git lfs pull。"
        )
    return metadata
