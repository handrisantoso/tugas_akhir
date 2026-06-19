"""Run Tekken player tracking, YOLO-pose keypoints, and ST-GCN move inference.

This script is intentionally close to the notebook pipeline:
YOLO detector -> OC-SORT player IDs -> cropped ROI YOLO-pose -> rolling ST-GCN++.
It loads the MMAction checkpoint directly, so exporting the classifier is not
needed for normal local testing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MMACTION_DIR = ROOT / "mmaction2"
if str(MMACTION_DIR) not in sys.path:
    sys.path.insert(0, str(MMACTION_DIR))

from boxmot.trackers.ocsort.ocsort import OcSort  # noqa: E402
from mmaction.apis import init_recognizer  # noqa: E402
from mmengine.dataset import Compose, pseudo_collate  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402


COCO_EDGES = (
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
)


@dataclass
class Prediction:
    label: str = "warming up"
    confidence: float = 0.0
    raw_label: str = ""


@dataclass
class PlayerState:
    name: str
    color: tuple[int, int, int]
    track_id: Optional[int] = None
    keypoints: deque[np.ndarray] = field(default_factory=deque)
    prediction: Prediction = field(default_factory=Prediction)
    pending_prediction: Prediction = field(default_factory=Prediction)
    pending_count: int = 0
    missing_frames: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tekken YOLO-pose + ST-GCN++ video inference demo."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "practice_videos" / "Bryan_LR_Complete.mp4",
        help="Input video path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "practice_videos" / "tekken_stgcn_demo.mp4",
        help="Output visualization video path.",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=ROOT / "runs" / "detect" / "practice_m" / "weights" / "best.pt",
        help="YOLO player detector weights.",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=ROOT / "yolo11l-pose.pt",
        help="YOLO pose weights. Use yolo11m-pose.pt for faster or yolo11x-pose.pt for stronger.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "mmaction2"
        / "configs"
        / "skeleton"
        / "stgcnpp"
        / "tekken_stgcn.py",
        help="MMAction2 ST-GCN++ config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "mmaction2"
        / "work_dirs"
        / "tekken_stgcn_v3_5move_val"
        / "best_acc_top1_epoch_10.pth",
        help="Trained ST-GCN++ checkpoint.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "practice_videos" / "move_labels_filtered.json",
        help="Label map JSON used by the filtered 5-move dataset.",
    )
    parser.add_argument("--device", default="mps", help="YOLO device, e.g. mps/cpu/cuda:0.")
    parser.add_argument(
        "--action-device",
        default="cpu",
        help="ST-GCN device. CPU is usually enough and avoids MPS checkpoint quirks.",
    )
    parser.add_argument("--det-conf", type=float, default=0.50, help="Detector confidence.")
    parser.add_argument("--det-iou", type=float, default=0.35, help="Detector NMS IoU.")
    parser.add_argument("--pose-conf", type=float, default=0.25, help="Pose confidence.")
    parser.add_argument("--pose-iou", type=float, default=0.35, help="Pose NMS IoU.")
    parser.add_argument("--pose-imgsz", type=int, default=384, help="YOLO-pose image size.")
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Pixels added around each tracked box before pose extraction.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Rolling keypoint window length sent to ST-GCN++.",
    )
    parser.add_argument(
        "--kp-interpolation",
        choices=("window", "none"),
        default="window",
        help=(
            "How to handle missing keypoints before ST-GCN++. "
            "'window' keeps the v10 behavior by interpolating inside the current rolling window; "
            "'none' zero-fills missing keypoints for the v12 no-interpolation test."
        ),
    )
    parser.add_argument(
        "--predict-every",
        type=int,
        default=3,
        help="Run ST-GCN once every N frames per player.",
    )
    parser.add_argument(
        "--action-conf",
        type=float,
        default=0.35,
        help="Display 'uncertain' when the top action probability is below this.",
    )
    parser.add_argument(
        "--stable-predictions",
        type=int,
        default=2,
        help="Only update the visible action after the same label appears this many prediction steps in a row.",
    )
    parser.add_argument(
        "--clear-missing-frames",
        type=int,
        default=8,
        help="Clear a player's keypoint buffer/prediction after this many consecutive missing tracked boxes.",
    )
    parser.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=0.015,
        help="Prefer tracked boxes above this frame-area ratio when choosing initial players.",
    )
    parser.add_argument(
        "--lost-reset-frames",
        type=int,
        default=45,
        help="Re-select player IDs if both assigned players disappear this long.",
    )
    parser.add_argument(
        "--character-name",
        default="",
        help="Optional character name shown in the large prediction panels. Empty means show P1/P2.",
    )
    parser.add_argument(
        "--strip-label-prefix",
        default="Bryan ",
        help="Prefix removed from displayed move labels. Use '' to show full class names.",
    )
    parser.add_argument(
        "--hide-labels",
        default="idle",
        help="Comma-separated labels to hide in the large panels, e.g. idle or idle,unknown.",
    )
    parser.add_argument(
        "--panel-alpha",
        type=float,
        default=0.58,
        help="Opacity of the bottom prediction panels.",
    )
    parser.add_argument(
        "--classify-players",
        choices=("both", "p1", "p2"),
        default="both",
        help="Choose which tracked player gets ST-GCN move predictions.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means full video.")
    parser.add_argument("--show", action="store_true", help="Show a live OpenCV window.")
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_labels(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_label: dict[int, str] = {}
    for key, value in label_map.items():
        if isinstance(value, int):
            id_to_label[value] = str(key)
        else:
            id_to_label[int(key)] = str(value)
    return dict(sorted(id_to_label.items()))


def make_tracker(device: str) -> OcSort:
    return OcSort(
        half=device != "cpu",
        device=device,
        max_age=90,
        min_hits=2,
        iou_threshold=0.15,
        det_thresh=0.20,
    )


def empty_keypoints() -> np.ndarray:
    return np.full((17, 3), np.nan, dtype=np.float32)


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.array([], dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def choose_pose_candidate(
    boxes_xyxy: np.ndarray, box_scores: np.ndarray, target_box: Optional[np.ndarray] = None
) -> int:
    if boxes_xyxy.shape[0] == 0:
        return 0
    if target_box is None:
        return int(np.argmax(box_scores))

    target_box = np.asarray(target_box, dtype=np.float32)
    ious = box_iou_xyxy(target_box, boxes_xyxy)

    target_center = np.array(
        [(target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2],
        dtype=np.float32,
    )
    candidate_centers = np.column_stack(
        ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2, (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2)
    )
    distances = np.linalg.norm(candidate_centers - target_center, axis=1)
    target_diag = max(
        1.0, np.linalg.norm([target_box[2] - target_box[0], target_box[3] - target_box[1]])
    )
    center_score = 1.0 - np.clip(distances / target_diag, 0.0, 1.0)

    match_score = (2.0 * ious) + center_score + (0.25 * box_scores)
    return int(np.argmax(match_score))


def yolo_detect(model: YOLO, frame: np.ndarray, conf: float, iou: float, device: str) -> np.ndarray:
    result = model(frame, conf=conf, iou=iou, classes=[0], device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy().reshape(-1, 1)
    cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.hstack((xyxy, scores, cls)).astype(np.float32)


def ocsort_tracking(tracker: OcSort, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
    if detections.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32)

    tracks = tracker.update(detections, frame)
    if tracks is None or len(tracks) == 0:
        return np.empty((0, 5), dtype=np.float32)

    ids = tracks.id.astype(np.float32).reshape(-1, 1)
    boxes = tracks.xyxy.astype(np.float32)
    return np.hstack((ids, boxes))


def pad_box(frame_h: int, frame_w: int, box: np.ndarray, padding: int) -> np.ndarray:
    obj_id, x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(frame_w, int(x2) + padding)
    y2 = min(frame_h, int(y2) + padding)
    return np.array([obj_id, x1, y1, x2, y2], dtype=np.float32)


def crop_roi(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    _, x1, y1, x2, y2 = box.astype(int)
    return frame[y1:y2, x1:x2]


def track_box_to_roi_box(track_box: np.ndarray, crop_box: np.ndarray) -> np.ndarray:
    _, tx1, ty1, tx2, ty2 = track_box
    _, cx1, cy1, _, _ = crop_box
    return np.array([tx1 - cx1, ty1 - cy1, tx2 - cx1, ty2 - cy1], dtype=np.float32)


def run_yolopose(
    model: YOLO,
    image: np.ndarray,
    input_size: int,
    conf: float,
    iou: float,
    device: str,
    target_box: Optional[np.ndarray] = None,
) -> np.ndarray:
    if image is None or image.size == 0:
        return empty_keypoints()

    roi_height, roi_width = image.shape[:2]
    if roi_height == 0 or roi_width == 0:
        return empty_keypoints()

    result = model(
        image,
        imgsz=input_size,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        return empty_keypoints()

    xy = result.keypoints.xy.cpu().numpy()
    kp_conf = result.keypoints.conf
    if kp_conf is None:
        scores = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        scores = kp_conf.cpu().numpy()

    if xy.shape[0] == 0:
        return empty_keypoints()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        box_scores = result.boxes.conf.cpu().numpy()
        best_idx = choose_pose_candidate(boxes_xyxy, box_scores, target_box)
        best_idx = min(best_idx, xy.shape[0] - 1)
    else:
        mean_scores = np.nanmean(scores, axis=1)
        best_idx = 0 if np.all(np.isnan(mean_scores)) else int(np.nanargmax(mean_scores))

    keypoints = empty_keypoints()
    points_xy = xy[best_idx]
    point_scores = scores[best_idx]
    num_points = min(17, points_xy.shape[0])

    keypoints[:num_points, 0] = points_xy[:num_points, 1] / roi_height
    keypoints[:num_points, 1] = points_xy[:num_points, 0] / roi_width
    keypoints[:num_points, 2] = point_scores[:num_points]
    return keypoints


def normalize_points_to_full_frame(
    kp_array: np.ndarray, box: np.ndarray, full_height: int, full_width: int
) -> np.ndarray:
    _, x1, y1, x2, y2 = box
    roi_height = max(1.0, y2 - y1)
    roi_width = max(1.0, x2 - x1)

    kp_full = empty_keypoints()
    for i, (y_norm, x_norm, score) in enumerate(kp_array):
        if not np.isfinite(y_norm) or not np.isfinite(x_norm):
            continue
        y_full = ((y_norm * roi_height) + y1) / full_height
        x_full = ((x_norm * roi_width) + x1) / full_width
        kp_full[i] = [y_full, x_full, score]
    return kp_full


def interpolate_window(kp_window: deque[np.ndarray]) -> np.ndarray:
    window = np.asarray(kp_window, dtype=np.float32).copy()
    if window.ndim != 3:
        raise ValueError(f"Expected keypoint window shape (T, 17, 3), got {window.shape}")

    frame_idx = np.arange(window.shape[0])
    for joint in range(window.shape[1]):
        for coord in range(2):
            data = window[:, joint, coord]
            mask = np.isfinite(data)
            if mask.any():
                window[:, joint, coord] = np.interp(frame_idx, frame_idx[mask], data[mask])
            else:
                window[:, joint, coord] = 0.0

        score = window[:, joint, 2]
        score[~np.isfinite(score)] = 0.0
        window[:, joint, 2] = score

    return window


def zero_fill_window(kp_window: deque[np.ndarray]) -> np.ndarray:
    window = np.asarray(kp_window, dtype=np.float32).copy()
    if window.ndim != 3:
        raise ValueError(f"Expected keypoint window shape (T, 17, 3), got {window.shape}")
    return np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_action_window(kp_window: deque[np.ndarray], kp_interpolation: str) -> np.ndarray:
    if kp_interpolation == "window":
        # v10 behavior: fill missing coordinates only inside the current 30-frame window.
        return interpolate_window(kp_window)

    # v12 behavior: preserve missing keypoints as missing, then encode them as zero + zero score.
    # window = interpolate_window(kp_window)
    return zero_fill_window(kp_window)


def build_action_sample(
    kp_window: deque[np.ndarray],
    frame_h: int,
    frame_w: int,
    kp_interpolation: str,
) -> dict:
    window = prepare_action_window(kp_window, kp_interpolation)
    keypoint_xy = window[:, :, [1, 0]][None].astype(np.float32)
    keypoint_score = window[:, :, 2][None].astype(np.float32)
    return {
        "frame_dir": "realtime_window",
        "label": -1,
        "img_shape": (frame_h, frame_w),
        "original_shape": (frame_h, frame_w),
        "total_frames": keypoint_xy.shape[1],
        "keypoint": keypoint_xy,
        "keypoint_score": keypoint_score,
    }


def init_action_model(config: Path, checkpoint: Path, device: str):
    model = init_recognizer(str(config), str(checkpoint), device=device)
    init_default_scope(model.cfg.get("default_scope", "mmaction"))
    pipeline = Compose(model.cfg.test_pipeline)
    model.eval()
    return model, pipeline


def softmax_if_needed(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32)
    if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=1e-3):
        return scores
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.maximum(exp_scores.sum(), 1e-12)


def predict_action(
    model,
    pipeline: Compose,
    kp_window: deque[np.ndarray],
    frame_h: int,
    frame_w: int,
    id_to_label: dict[int, str],
    confidence_threshold: float,
    kp_interpolation: str,
) -> Prediction:
    sample = build_action_sample(kp_window, frame_h, frame_w, kp_interpolation)
    data = pipeline(sample)
    data = pseudo_collate([data])

    with torch.no_grad():
        result = model.test_step(data)[0]

    scores = result.pred_score.detach().cpu().numpy()
    probs = softmax_if_needed(scores)
    pred_idx = int(np.argmax(probs))
    raw_label = id_to_label.get(pred_idx, f"class_{pred_idx}")
    confidence = float(probs[pred_idx])
    label = raw_label if confidence >= confidence_threshold else "uncertain"
    return Prediction(label=label, confidence=confidence, raw_label=raw_label)


def should_classify_player(player: PlayerState, mode: str) -> bool:
    return mode == "both" or (mode == "p1" and player.name == "P1") or (
        mode == "p2" and player.name == "P2"
    )


def reset_player_prediction(player: PlayerState, label: str = "warming up") -> None:
    player.keypoints.clear()
    player.prediction = Prediction(label=label)
    player.pending_prediction = Prediction(label=label)
    player.pending_count = 0


def update_stable_prediction(player: PlayerState, new_prediction: Prediction, stable_steps: int) -> None:
    stable_steps = max(1, stable_steps)
    if new_prediction.label == "uncertain":
        return

    if new_prediction.label == player.pending_prediction.label:
        player.pending_count += 1
    else:
        player.pending_prediction = new_prediction
        player.pending_count = 1

    if player.pending_count >= stable_steps:
        player.prediction = new_prediction


def select_initial_player_ids(
    id_box_array: np.ndarray, frame_h: int, frame_w: int, min_area_ratio: float
) -> tuple[Optional[int], Optional[int]]:
    if id_box_array.shape[0] < 2:
        return None, None

    boxes = id_box_array[:, 1:5]
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    min_area = frame_h * frame_w * min_area_ratio
    candidate_indices = np.where(areas >= min_area)[0]
    if candidate_indices.size < 2:
        candidate_indices = np.argsort(-areas)[:2]
    else:
        candidate_indices = candidate_indices[np.argsort(-areas[candidate_indices])[:2]]

    candidates = id_box_array[candidate_indices]
    centers_x = (candidates[:, 1] + candidates[:, 3]) / 2
    order = np.argsort(centers_x)
    return int(candidates[order[0], 0]), int(candidates[order[1], 0])


def find_track_box(id_box_array: np.ndarray, track_id: Optional[int]) -> Optional[np.ndarray]:
    if track_id is None or id_box_array.size == 0:
        return None
    matches = id_box_array[id_box_array[:, 0] == track_id]
    if matches.size == 0:
        return None
    return matches[0]


def extract_player_keypoints(
    pose_model: YOLO,
    frame: np.ndarray,
    track_box: Optional[np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    frame_h, frame_w = frame.shape[:2]
    if track_box is None:
        return empty_keypoints(), None

    padded_box = pad_box(frame_h, frame_w, track_box, args.padding)
    roi = crop_roi(frame, padded_box)
    target_box = track_box_to_roi_box(track_box, padded_box)
    kp_raw = run_yolopose(
        pose_model,
        roi,
        input_size=args.pose_imgsz,
        conf=args.pose_conf,
        iou=args.pose_iou,
        device=args.device,
        target_box=target_box,
    )
    kp_full = normalize_points_to_full_frame(kp_raw, padded_box, frame_h, frame_w)
    return kp_full, padded_box


def draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: tuple[int, int, int],
    score_threshold: float = 0.05,
) -> None:
    frame_h, frame_w = frame.shape[:2]

    for a, b in COCO_EDGES:
        ya, xa, ca = keypoints[a]
        yb, xb, cb = keypoints[b]
        if not all(np.isfinite(v) for v in (ya, xa, yb, xb)):
            continue
        if (np.isfinite(ca) and ca < score_threshold) or (np.isfinite(cb) and cb < score_threshold):
            continue
        pt_a = (int(np.clip(xa * frame_w, 0, frame_w - 1)), int(np.clip(ya * frame_h, 0, frame_h - 1)))
        pt_b = (int(np.clip(xb * frame_w, 0, frame_w - 1)), int(np.clip(yb * frame_h, 0, frame_h - 1)))
        cv2.line(frame, pt_a, pt_b, color, 2, lineType=cv2.LINE_AA)

    for y, x, score in keypoints:
        if not np.isfinite(y) or not np.isfinite(x):
            continue
        if np.isfinite(score) and score < score_threshold:
            continue
        point = (int(np.clip(x * frame_w, 0, frame_w - 1)), int(np.clip(y * frame_h, 0, frame_h - 1)))
        cv2.circle(frame, point, 3, color, thickness=-1, lineType=cv2.LINE_AA)


def draw_box_and_label(
    frame: np.ndarray,
    box: Optional[np.ndarray],
    player: PlayerState,
) -> None:
    if box is None:
        return

    obj_id, x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), player.color, 2, lineType=cv2.LINE_AA)

    text = f"{player.name} ID:{obj_id}"
    text_y = max(24, y1 - 8)
    cv2.putText(
        frame,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        player.color,
        2,
        lineType=cv2.LINE_AA,
    )


def fit_text_scale(
    text: str,
    max_width: int,
    font_face: int,
    initial_scale: float,
    thickness: int,
    min_scale: float = 0.45,
) -> float:
    scale = initial_scale
    while scale > min_scale:
        text_width = cv2.getTextSize(text, font_face, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale -= 0.05
    return min_scale


def draw_translucent_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)


def display_move_label(label: str, strip_prefix: str) -> str:
    if strip_prefix and label.startswith(strip_prefix):
        return label[len(strip_prefix):].strip()
    return label


def parse_hidden_labels(hidden_labels: str) -> set[str]:
    return {item.strip().lower() for item in hidden_labels.split(",") if item.strip()}


def draw_prediction_panels(
    frame: np.ndarray,
    players: list[PlayerState],
    character_name: str,
    strip_label_prefix: str,
    hidden_labels: set[str],
    alpha: float,
) -> None:
    frame_h, frame_w = frame.shape[:2]
    margin_x = max(24, int(frame_w * 0.025))
    panel_gap = max(28, int(frame_w * 0.08))
    panel_w = int((frame_w - (2 * margin_x) - panel_gap) / 2)
    panel_h = max(120, int(frame_h * 0.24))
    panel_y1 = frame_h - panel_h - max(18, int(frame_h * 0.045))
    panel_y2 = panel_y1 + panel_h

    panel_positions = [
        (margin_x, panel_y1, margin_x + panel_w, panel_y2),
        (frame_w - margin_x - panel_w, panel_y1, frame_w - margin_x, panel_y2),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    for player, (x1, y1, x2, y2) in zip(players, panel_positions):
        draw_translucent_rect(frame, x1, y1, x2, y2, (35, 35, 35), alpha)

        pad_x = max(20, int(panel_w * 0.04))
        title = character_name if character_name else player.name
        raw_move = player.prediction.label
        move = display_move_label(raw_move, strip_label_prefix)
        should_hide = raw_move.lower() in hidden_labels or move.lower() in hidden_labels
        if should_hide:
            move = ""
        elif player.prediction.confidence > 0:
            move = f"{move}  {player.prediction.confidence:.2f}"

        title_scale = fit_text_scale(title, panel_w - 2 * pad_x, font, 1.35, 2)
        move_scale = fit_text_scale(move, panel_w - 2 * pad_x, font, 1.20, 2)

        title_y = y1 + int(panel_h * 0.27)
        move_y = y1 + int(panel_h * 0.58)
        cv2.putText(
            frame,
            title,
            (x1 + pad_x, title_y),
            font,
            title_scale,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        if move:
            cv2.putText(
                frame,
                move,
                (x1 + pad_x, move_y),
                font,
                move_scale,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )


def draw_status(
    frame: np.ndarray,
    frame_idx: int,
    timings_ms: dict[str, float],
    window_size: int,
) -> None:
    text = (
        f"frame {frame_idx} | "
        f"det {timings_ms['det']:.2f}ms | "
        f"pose {timings_ms['pose']:.2f}ms | "
        f"action {timings_ms['action']:.2f}ms | "
        f"total {timings_ms['total']:.2f}ms | "
        f"window {window_size}"
    )
    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )


def process_video(args: argparse.Namespace) -> None:
    require_path(args.video, "Input video")
    require_path(args.detector, "Detector weights")
    require_path(args.pose_model, "Pose weights")
    require_path(args.config, "MMAction config")
    require_path(args.checkpoint, "ST-GCN checkpoint")
    require_path(args.labels, "Label map")

    id_to_label = load_labels(args.labels)
    detector = YOLO(str(args.detector))
    pose_model = YOLO(str(args.pose_model))
    action_model, action_pipeline = init_action_model(args.config, args.checkpoint, args.action_device)
    tracker = make_tracker(args.device)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    players = [
        PlayerState(name="P1", color=(0, 255, 0), keypoints=deque(maxlen=args.window_size)),
        PlayerState(name="P2", color=(255, 60, 0), keypoints=deque(maxlen=args.window_size)),
    ]
    player_set = False
    both_missing_frames = 0
    frame_idx = 0

    print(f"Input: {args.video}")
    print(f"Output: {args.output}")
    print(f"Detector: {args.detector}")
    print(f"Pose: {args.pose_model} imgsz={args.pose_imgsz}")
    print(f"ST-GCN: {args.checkpoint}")
    print(f"KP interpolation: {args.kp_interpolation}")
    print(f"Labels: {id_to_label}")

    while True:
        if args.max_frames and frame_idx >= args.max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame_start = time.perf_counter()
        timings_ms = {"det": 0.0, "pose": 0.0, "action": 0.0, "total": 0.0}

        det_start = time.perf_counter()
        detections = yolo_detect(detector, frame, args.det_conf, args.det_iou, args.device)
        id_box_array = ocsort_tracking(tracker, detections, frame)
        timings_ms["det"] = (time.perf_counter() - det_start) * 1000

        if not player_set and id_box_array.shape[0] >= 2:
            p1_id, p2_id = select_initial_player_ids(
                id_box_array, frame_h, frame_w, args.min_box_area_ratio
            )
            if p1_id is not None and p2_id is not None:
                players[0].track_id = p1_id
                players[1].track_id = p2_id
                player_set = True

        player_boxes = [find_track_box(id_box_array, player.track_id) for player in players]
        if player_set and player_boxes[0] is None and player_boxes[1] is None:
            both_missing_frames += 1
            if both_missing_frames > args.lost_reset_frames:
                tracker = make_tracker(args.device)
                for player in players:
                    player.track_id = None
                    reset_player_prediction(player)
                player_set = False
                both_missing_frames = 0
        else:
            both_missing_frames = 0

        pose_start = time.perf_counter()
        padded_boxes: list[Optional[np.ndarray]] = []
        for player, track_box in zip(players, player_boxes):
            if track_box is None:
                player.missing_frames += 1
                if player.missing_frames >= args.clear_missing_frames:
                    reset_player_prediction(player, label="")
            else:
                player.missing_frames = 0

            kp_full, padded_box = extract_player_keypoints(pose_model, frame, track_box, args)
            player.keypoints.append(kp_full)
            padded_boxes.append(padded_box)
        timings_ms["pose"] = (time.perf_counter() - pose_start) * 1000

        action_start = time.perf_counter()
        if frame_idx % max(1, args.predict_every) == 0:
            for player in players:
                if not should_classify_player(player, args.classify_players):
                    player.prediction = Prediction(label="not evaluated")
                    continue
                if len(player.keypoints) == args.window_size:
                    new_prediction = predict_action(
                        action_model,
                        action_pipeline,
                        player.keypoints,
                        frame_h,
                        frame_w,
                        id_to_label,
                        args.action_conf,
                        args.kp_interpolation,
                    )
                    update_stable_prediction(player, new_prediction, args.stable_predictions)
        timings_ms["action"] = (time.perf_counter() - action_start) * 1000

        for player, padded_box in zip(players, padded_boxes):
            draw_box_and_label(frame, padded_box, player)
            if len(player.keypoints) > 0:
                draw_keypoints(frame, player.keypoints[-1], player.color)

        timings_ms["total"] = (time.perf_counter() - frame_start) * 1000
        draw_prediction_panels(
            frame,
            players,
            args.character_name,
            args.strip_label_prefix,
            parse_hidden_labels(args.hide_labels),
            args.panel_alpha,
        )
        draw_status(frame, frame_idx, timings_ms, min(len(players[0].keypoints), args.window_size))
        writer.write(frame)

        if args.show:
            cv2.imshow("Tekken ST-GCN inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 100 == 0:
            print(
                f"frame {frame_idx:5d} | "
                f"det {timings_ms['det']:.2f} ms | "
                f"pose {timings_ms['pose']:.2f} ms | "
                f"action {timings_ms['action']:.2f} ms | "
                f"total {timings_ms['total']:.2f} ms"
            )

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. Wrote: {args.output}")


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
