from __future__ import annotations
import contextlib
import io
import os
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageSequence

from app.config import settings
from app.utils.helpers import ndarray_to_base64


POLICE_GESTURES = {
    0: ("no_gesture", "无手势"),
    1: ("stop", "停止"),
    2: ("go_straight", "直行"),
    3: ("turn_left", "左转弯"),
    4: ("left_turn_wait", "左转弯待转"),
    5: ("turn_right", "右转弯"),
    6: ("lane_change", "变道"),
    7: ("slow_down", "减速慢行"),
    8: ("pull_over", "靠边停车"),
}

CTPGR_POSE_CONNECTIONS = [
    (1, 2), (2, 3), (4, 5), (5, 6),
    (14, 1), (14, 4), (1, 7), (4, 10),
    (7, 8), (8, 9), (10, 11), (11, 12), (13, 14),
]


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class PoliceGestureService:
    def __init__(self):
        self.ctpgr_root = settings.base_dir.parent / "ctpgr-pytorch-master"
        self.input_size = (512, 512)
        self.sequence_steps = 30
        self._predictor = None
        self._pg = None
        self._model_lock = threading.RLock()

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = self._load_ctpgr_predictor()
        return self._predictor

    @property
    def pg(self):
        if self._pg is None:
            self._predictor = self._load_ctpgr_predictor()
        return self._pg

    def _load_ctpgr_predictor(self):
        if not self.ctpgr_root.exists():
            raise FileNotFoundError(f"ctpgr project not found: {self.ctpgr_root}")
        checkpoints = self.ctpgr_root / "checkpoints"
        missing = [name for name in ("pose_model.pt", "lstm.pt") if not (checkpoints / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing ctpgr checkpoints: {', '.join(missing)}")
        root_str = str(self.ctpgr_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        with _working_directory(self.ctpgr_root):
            from constants.enum_keys import PG
            from pred.gesture_pred import GesturePred
            self._pg = PG
            return GesturePred()

    def _detect_best_frame(self, image_bytes: bytes) -> np.ndarray:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            if getattr(pil_img, "is_animated", False):
                best_frame = None
                best_score = -1.0
                for frame in ImageSequence.Iterator(pil_img):
                    frame_np = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
                    score = cv2.Laplacian(cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if score > best_score:
                        best_score = score
                        best_frame = frame_np
                if best_frame is not None:
                    return best_frame
            return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to parse image")
        return image

    def _confidence(self, scores: np.ndarray, gesture_id: int) -> float:
        probs = torch.softmax(torch.from_numpy(scores.astype(np.float32)), dim=0).numpy()
        if 0 <= gesture_id < len(probs):
            return float(probs[gesture_id])
        return 0.0

    def create_sequence_state(self) -> dict[str, torch.Tensor]:
        return {
            "h": torch.zeros_like(self.predictor.g_model.h0()),
            "c": torch.zeros_like(self.predictor.g_model.c0()),
        }

    def reset_sequence_state(self) -> None:
        self.predictor.h = torch.zeros_like(self.predictor.h)
        self.predictor.c = torch.zeros_like(self.predictor.c)

    def _extract_keypoints(self, coord_norm: np.ndarray) -> list[dict]:
        if coord_norm.ndim == 3:
            coord_norm = coord_norm[0]
        if coord_norm.shape[0] != 2:
            coord_norm = coord_norm.T
        w, h = self.input_size
        return [
            {"id": i + 1, "x": round(float(coord_norm[0, i] * w), 2), "y": round(float(coord_norm[1, i] * h), 2), "z": 0.0, "visibility": 1.0}
            for i in range(coord_norm.shape[1])
        ]

    def _draw_skeleton(self, image: np.ndarray, keypoints: list[dict]) -> None:
        points = {p["id"]: (int(p["x"]), int(p["y"])) for p in keypoints}
        for a, b in CTPGR_POSE_CONNECTIONS:
            if a in points and b in points:
                cv2.line(image, points[a], points[b], (0, 255, 0), 2)
        for point in points.values():
            cv2.circle(image, point, 4, (0, 200, 255), -1)

    def _result_payload(self, ctpgr_image: np.ndarray, result) -> dict[str, Any]:
        gesture_id = int(result[self.pg.OUT_ARGMAX])
        scores = result[self.pg.OUT_SCORES]
        confidence = self._confidence(scores, gesture_id)
        keypoints = self._extract_keypoints(result[self.pg.COORD_NORM])
        annotated = ctpgr_image.copy()
        self._draw_skeleton(annotated, keypoints)
        en, cn = POLICE_GESTURES.get(gesture_id, POLICE_GESTURES[0])
        cv2.putText(annotated, f"{en} ({confidence:.0%})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return {
            "gesture": en,
            "gesture_cn": cn,
            "gesture_id": gesture_id,
            "confidence": round(confidence, 3),
            "keypoints": keypoints,
            "annotated_image": ndarray_to_base64(annotated),
            "success": gesture_id > 0,
        }

    def _coord_from_prepared_image(self, ctpgr_image: np.ndarray) -> np.ndarray:
        with self._model_lock:
            pose = self.predictor.p_predictor.get_coordinates(ctpgr_image)
        return pose[self.pg.COORD_NORM][np.newaxis]

    def _classify_coord(self, coord_norm: np.ndarray, state: dict[str, torch.Tensor] | None = None):
        features_dict = self.predictor.bla.handcrafted_features(coord_norm)
        features = np.concatenate(
            (
                features_dict[self.pg.BONE_LENGTH],
                features_dict[self.pg.BONE_ANGLE_COS],
                features_dict[self.pg.BONE_ANGLE_SIN],
            ),
            axis=1,
        )
        features = features[np.newaxis].transpose((1, 0, 2))
        features = torch.from_numpy(features).to(self.predictor.g_model.device, dtype=torch.float32)
        if state is None:
            state = self.create_sequence_state()
        with self._model_lock, torch.no_grad():
            _, h, c, class_out = self.predictor.g_model(features, state["h"], state["c"])
        state["h"], state["c"] = h, c
        scores = class_out[0].cpu().numpy()
        return {self.pg.OUT_ARGMAX: int(np.argmax(scores)), self.pg.OUT_SCORES: scores, self.pg.COORD_NORM: coord_norm}

    def _classify_prepared_image(self, ctpgr_image: np.ndarray) -> dict[str, Any]:
        coord_norm = self._coord_from_prepared_image(ctpgr_image)
        state = self.create_sequence_state()

        sequence_results = []
        for _ in range(self.sequence_steps):
            sequence_results.append(self._classify_coord(coord_norm, state))

        tail = sequence_results[-8:]
        nonzero_tail = [r for r in tail if int(r[self.pg.OUT_ARGMAX]) > 0]
        if nonzero_tail:
            result = max(nonzero_tail, key=lambda r: self._confidence(r[self.pg.OUT_SCORES], int(r[self.pg.OUT_ARGMAX])))
        else:
            result = sequence_results[-1]
        return self._result_payload(ctpgr_image, result)

    def recognize_prepared_frame_continuous(self, ctpgr_image: np.ndarray, state: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        coord_norm = self._coord_from_prepared_image(ctpgr_image)
        result = self._classify_coord(coord_norm, state)
        return self._result_payload(ctpgr_image, result)

    def recognize_image(self, image: np.ndarray) -> dict[str, Any]:
        ctpgr_image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        return self._classify_prepared_image(ctpgr_image)

    def recognize(self, image_bytes: bytes) -> dict[str, Any]:
        image = self._detect_best_frame(image_bytes)
        return self.recognize_image(image)

    def recognize_frame(self, frame: np.ndarray) -> dict[str, Any]:
        return self.recognize_image(frame)

    def recognize_frame_continuous(self, frame: np.ndarray, state: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        ctpgr_image = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
        return self.recognize_prepared_frame_continuous(ctpgr_image, state)


police_gesture_service = PoliceGestureService()
