"""Search LSTM label alignment and class weights without touching runtime weights.

Candidates are selected using validation videos only.  The independent test
split is opened only after the winning validation configuration is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from train_yolo_pose_gesture_model import ROOT, evaluate_checkpoint, evaluate_model, load_cache, train
from models.gesture_recognition_model import GestureRecognitionModel


BASELINE_PATH = ROOT / "checkpoints" / "lstm_yolo11s.pt"
OPTIMIZATION_ROOT = ROOT / "generated" / "yolo11s_lstm_optimization"
CANDIDATE_ROOT = ROOT / "checkpoints" / "yolo11s_lstm_candidates"
SELECTED_PATH = ROOT / "checkpoints" / "lstm_yolo11s_validation_winner.pt"
SUMMARY_PATH = OPTIMIZATION_ROOT / "summary.json"

WEIGHT_CONFIGS = (
    ("uniform", 0.0, 1.0),
    ("mild", 0.25, 1.0),
    ("sqrt", 0.5, 1.0),
    ("sqrt_ng15", 0.5, 1.5),
)
LABEL_DELAYS = (0, 5, 10, 15, 20)


def candidate_name(weight_name: str, delay: int) -> str:
    return f"{weight_name}_delay{delay:02d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-len", type=int, default=450)
    parser.add_argument("--stride", type=int, default=225)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-videos", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--comparison-label-delay",
        type=int,
        default=15,
        help="fixed label alignment used to compare every candidate",
    )
    parser.add_argument("--force", action="store_true", help="retrain candidates whose reports already exist")
    return parser.parse_args()


def run_candidate(args, weight_name: str, power: float, multiplier: float, delay: int) -> dict:
    name = candidate_name(weight_name, delay)
    comparison_dir = (
        f"v2_weightnorm_compare_delay{args.comparison_label_delay:02d}"
        f"_seed{args.seed}_epochs{args.epochs}_clip{args.clip_len}_stride{args.stride}"
    )
    model_path = CANDIDATE_ROOT / comparison_dir / f"{name}.pt"
    report_path = OPTIMIZATION_ROOT / comparison_dir / f"{name}.json"
    if model_path.is_file() and report_path.is_file() and not args.force:
        print(f"cached candidate: {name}", flush=True)
        return json.loads(report_path.read_text(encoding="utf-8"))

    print(f"training candidate: {name}", flush=True)
    train_args = SimpleNamespace(
        batch_size=args.batch_size,
        clip_len=args.clip_len,
        stride=args.stride,
        epochs=args.epochs,
        patience=args.patience,
        validation_videos=args.validation_videos,
        learning_rate=args.learning_rate,
        seed=args.seed,
        label_delay=delay,
        validation_label_delay=args.comparison_label_delay,
        class_weight_power=power,
        no_gesture_weight_multiplier=multiplier,
        output_path=model_path,
        report_path=report_path,
        skip_test=True,
        device=args.device,
    )
    return train(train_args)


def main() -> None:
    args = parse_args()
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(f"baseline model not found: {BASELINE_PATH}")
    baseline_hash_before = sha256(BASELINE_PATH)
    OPTIMIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = []
    for weight_name, power, multiplier in WEIGHT_CONFIGS:
        for delay in LABEL_DELAYS:
            report = run_candidate(args, weight_name, power, multiplier, delay)
            training_alignment_validation = report["validation"]
            candidates.append(
                {
                    "name": candidate_name(weight_name, delay),
                    "model": report["model"],
                    "configuration": report["configuration"],
                    "training_alignment_validation": training_alignment_validation,
                }
            )

    # Scores measured against different label delays are not comparable.  Load
    # one fixed validation target and score every checkpoint against it before
    # selecting a winner.  The independent test split is still unopened here.
    fixed_validation = load_cache("train", args.comparison_label_delay)[-args.validation_videos :]
    device = torch.device(args.device)
    for candidate in candidates:
        model = GestureRecognitionModel(1).to(device)
        model.load_state_dict(torch.load(candidate["model"], map_location=device, weights_only=True))
        candidate["validation"] = evaluate_model(model, fixed_validation, device)

    winner = max(
        candidates,
        key=lambda item: (
            item["validation"]["macro_f1"],
            -item["validation"]["false_gesture_rate"],
        ),
    )
    winner_test = evaluate_checkpoint(Path(winner["model"]), args.device, args.comparison_label_delay)
    baseline_test = evaluate_checkpoint(BASELINE_PATH, args.device, args.comparison_label_delay)
    shutil.copy2(winner["model"], SELECTED_PATH)
    baseline_hash_after = sha256(BASELINE_PATH)
    if baseline_hash_after != baseline_hash_before:
        raise RuntimeError("runtime baseline changed during parameter search")

    summary = {
        "selection_split": "validation",
        "selection_metric": "fixed-alignment macro_f1; false_gesture_rate tie-breaker",
        "comparison_label_delay": args.comparison_label_delay,
        "test_opened_after_selection": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "winner": {**winner, "test": winner_test},
        "baseline_on_winner_test_labels": baseline_test,
        "selected_validation_model": str(SELECTED_PATH),
        "runtime_model": str(BASELINE_PATH),
        "runtime_model_sha256_before": baseline_hash_before,
        "runtime_model_sha256_after": baseline_hash_after,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "winner": winner["name"],
                "validation_macro_f1": winner["validation"]["macro_f1"],
                "winner_test_macro_f1": winner_test["macro_f1"],
                "baseline_test_macro_f1": baseline_test["macro_f1"],
                "summary": str(SUMMARY_PATH),
                "selected_validation_model": str(SELECTED_PATH),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
