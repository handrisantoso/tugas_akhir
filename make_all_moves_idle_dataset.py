"""Build an all-annotated-moves + idle ST-GCN dataset.

Positive clips come directly from Bryan_LR_Complete.json and the full
player keypoint arrays. Idle clips are sampled only from low-motion windows
that do not overlap any annotated move for that player.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "practice_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an all-moves + idle skeleton dataset.")
    parser.add_argument("--annotations", type=Path, default=DATA_DIR / "Bryan_LR_Complete.json")
    parser.add_argument("--player1-kp", type=Path, default=DATA_DIR / "player1_kp.npy")
    parser.add_argument("--player2-kp", type=Path, default=DATA_DIR / "player2_kp.npy")
    parser.add_argument("--output-dataset", type=Path, default=DATA_DIR / "skeleton_dataset_all_moves_idle.pkl")
    parser.add_argument("--output-labels", type=Path, default=DATA_DIR / "move_labels_all_moves_idle.json")
    parser.add_argument(
        "--positive-window-size",
        type=int,
        default=30,
        help="If >0, use a fixed-size context window around each annotated move instead of raw start/end frames.",
    )
    parser.add_argument(
        "--positive-window-mode",
        choices=("center", "end", "start"),
        default="center",
        help="How to align fixed positive windows when --positive-window-size is used.",
    )
    parser.add_argument(
        "--positive-offsets",
        default="-3,0,6",
        help="Comma-separated frame offsets for fixed positive windows, e.g. -6,0,6.",
    )
    parser.add_argument("--idle-label", default="idle")
    parser.add_argument("--idle-count", type=int, default=30)
    parser.add_argument("--idle-window-size", type=int, default=30)
    parser.add_argument("--idle-stride", type=int, default=15)
    parser.add_argument("--exclude-margin", type=int, default=15)
    parser.add_argument("--min-mean-score", type=float, default=0.35)
    parser.add_argument("--idle-pool-multiplier", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-height", type=int, default=720)
    parser.add_argument("--img-width", type=int, default=1280)
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def keypoints_to_annotation(
    name: str,
    label: int,
    keypoints_yxc: np.ndarray,
    img_shape: tuple[int, int],
    source_frame_dir: str | None = None,
) -> dict:
    coords_xy = keypoints_yxc[:, :, [1, 0]].astype(np.float32)
    scores = keypoints_yxc[:, :, 2].astype(np.float32)
    coords_xy = np.nan_to_num(coords_xy, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    ann = {
        "frame_dir": name,
        "label": label,
        "img_shape": img_shape,
        "original_shape": img_shape,
        "total_frames": int(keypoints_yxc.shape[0]),
        "keypoint": coords_xy[None],
        "keypoint_score": scores[None],
    }
    if source_frame_dir is not None:
        ann["source_frame_dir"] = source_frame_dir
    return ann


def fixed_window_bounds(
    start: int,
    end: int,
    total_frames: int,
    window_size: int,
    mode: str,
    offset: int = 0,
) -> tuple[int, int]:
    if window_size <= 0:
        return start, end
    if window_size >= total_frames:
        return 0, total_frames

    if mode == "start":
        win_start = start
    elif mode == "end":
        win_start = end - window_size
    else:
        center = (start + end) // 2
        win_start = center - (window_size // 2)

    win_start += offset
    win_start = max(0, min(win_start, total_frames - window_size))
    return win_start, win_start + window_size


def parse_offsets(offsets: str) -> list[int]:
    parsed = []
    for item in offsets.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    return parsed or [0]


def ordered_label_map(full_annotations: dict, idle_label: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    for item in full_annotations.values():
        label = f"{item['character']} {item['move']}"
        if label not in labels:
            labels[label] = len(labels)
    labels[idle_label] = len(labels)
    return labels


def build_move_block_masks(
    full_annotations: dict,
    total_frames: int,
    margin: int,
) -> dict[str, np.ndarray]:
    blocked = {
        "player1": np.zeros(total_frames, dtype=bool),
        "player2": np.zeros(total_frames, dtype=bool),
    }

    for item in full_annotations.values():
        player = item["player"]
        if player not in blocked:
            continue
        start = max(0, int(item["start_frame"]) - margin)
        end = min(total_frames - 1, int(item["end_frame"]) + margin)
        blocked[player][start : end + 1] = True
    return blocked


def window_motion(keypoints_yxc: np.ndarray) -> float:
    coords = keypoints_yxc[:, :, :2]
    scores = keypoints_yxc[:, :, 2]
    valid = np.isfinite(coords).all(axis=2) & np.isfinite(scores) & (scores > 0.3)
    diffs = np.linalg.norm(np.diff(np.nan_to_num(coords, nan=0.0), axis=0), axis=2)
    valid_pairs = valid[1:] & valid[:-1]
    if not valid_pairs.any():
        return float("inf")
    return float(np.nanmean(diffs[valid_pairs]))


def collect_idle_candidates(
    player_name: str,
    keypoints: np.ndarray,
    blocked: np.ndarray,
    window_size: int,
    stride: int,
    min_mean_score: float,
) -> list[tuple[str, int, int, float, float]]:
    candidates = []
    total_frames = keypoints.shape[0]

    for start in range(0, total_frames - window_size + 1, stride):
        end = start + window_size
        if blocked[start:end].any():
            continue

        window = keypoints[start:end]
        coords = window[:, :, :2]
        scores = window[:, :, 2]
        finite_ratio = float(np.isfinite(coords).mean())
        mean_score = float(np.nanmean(scores))
        if finite_ratio < 0.95 or mean_score < min_mean_score:
            continue

        motion = window_motion(window)
        if not np.isfinite(motion):
            continue
        candidates.append((player_name, start, end, mean_score, motion))

    return candidates


def split_by_class(
    annotations: list[dict],
    val_ratio: float,
    rng: random.Random,
) -> dict[str, list[str]]:
    by_label_group: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        group = ann.get("source_frame_dir", ann["frame_dir"])
        by_label_group[int(ann["label"])][group].append(ann["frame_dir"])

    train: list[str] = []
    val: list[str] = []
    for grouped_names in by_label_group.values():
        groups = list(grouped_names.items())
        rng.shuffle(groups)
        val_count = max(1, round(len(groups) * val_ratio)) if len(groups) > 1 else 0
        for _, names in groups[:val_count]:
            val.extend(names)
        for _, names in groups[val_count:]:
            train.extend(names)

    rng.shuffle(train)
    rng.shuffle(val)
    return {"train": train, "val": val}


def main() -> None:
    args = parse_args()
    for path, label in [
        (args.annotations, "Annotation JSON"),
        (args.player1_kp, "Player 1 keypoints"),
        (args.player2_kp, "Player 2 keypoints"),
    ]:
        require_path(path, label)

    rng = random.Random(args.seed)
    full_annotations = load_json(args.annotations)
    label_map = ordered_label_map(full_annotations, args.idle_label)
    idle_label_id = label_map[args.idle_label]
    img_shape = (args.img_height, args.img_width)
    positive_offsets = parse_offsets(args.positive_offsets)

    player_keypoints = {
        "player1": np.load(args.player1_kp).astype(np.float32),
        "player2": np.load(args.player2_kp).astype(np.float32),
    }
    total_frames = min(kp.shape[0] for kp in player_keypoints.values())

    output_annotations: list[dict] = []
    for name, item in full_annotations.items():
        player = item["player"]
        if player not in player_keypoints:
            continue

        raw_start = int(item["start_frame"])
        raw_end = int(item["end_frame"]) + 1
        label = f"{item['character']} {item['move']}"
        seen_windows = set()
        offsets = positive_offsets if args.positive_window_size > 0 else [0]
        for offset in offsets:
            start, end = fixed_window_bounds(
                raw_start,
                raw_end,
                player_keypoints[player].shape[0],
                args.positive_window_size,
                args.positive_window_mode,
                offset,
            )
            if start < 0 or end > player_keypoints[player].shape[0] or end <= start:
                continue
            if (start, end) in seen_windows:
                continue
            seen_windows.add((start, end))

            frame_dir = name
            if args.positive_window_size > 0:
                frame_dir = f"{name}_off{offset:+d}_w{start:05d}_{end - 1:05d}"
            output_annotations.append(
                keypoints_to_annotation(
                    frame_dir,
                    label_map[label],
                    player_keypoints[player][start:end],
                    img_shape,
                    source_frame_dir=name,
                )
            )

    blocked = build_move_block_masks(full_annotations, total_frames, args.exclude_margin)
    idle_candidates = []
    for player_name, keypoints in player_keypoints.items():
        idle_candidates.extend(
            collect_idle_candidates(
                player_name,
                keypoints,
                blocked[player_name],
                args.idle_window_size,
                args.idle_stride,
                args.min_mean_score,
            )
        )

    idle_candidates.sort(key=lambda item: item[4])
    pool_size = max(args.idle_count, args.idle_count * args.idle_pool_multiplier)
    idle_pool = idle_candidates[:pool_size]
    rng.shuffle(idle_pool)
    chosen_idle = idle_pool[: args.idle_count]

    if len(chosen_idle) < args.idle_count:
        print(f"Warning: requested {args.idle_count} idle clips, found {len(chosen_idle)}.")

    for idx, (player_name, start, end, _, motion) in enumerate(chosen_idle, start=1):
        name = f"{player_name}_idle_{idx:03d}_f{start:05d}_{end - 1:05d}_m{motion:.4f}"
        output_annotations.append(
            keypoints_to_annotation(
                name,
                idle_label_id,
                player_keypoints[player_name][start:end],
                img_shape,
                source_frame_dir=name,
            )
        )

    dataset = {
        "split": split_by_class(output_annotations, args.val_ratio, rng),
        "annotations": output_annotations,
    }

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    with args.output_dataset.open("wb") as f:
        pickle.dump(dataset, f)
    with args.output_labels.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
        f.write("\n")

    id_to_label = {idx: label for label, idx in label_map.items()}
    counts = Counter(int(ann["label"]) for ann in output_annotations)
    print(f"Saved dataset: {args.output_dataset}")
    print(f"Saved labels:  {args.output_labels}")
    print(f"Annotated clips: {len(output_annotations) - len(chosen_idle)}")
    print(f"Idle candidates: {len(idle_candidates)}")
    print(f"Idle chosen:     {len(chosen_idle)}")
    print(f"Train/val sizes: {len(dataset['split']['train'])}/{len(dataset['split']['val'])}")
    print("Class counts:")
    for label_id in sorted(counts):
        print(f"  {label_id:2d}: {id_to_label[label_id]} = {counts[label_id]}")


if __name__ == "__main__":
    main()
