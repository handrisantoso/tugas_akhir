#!/usr/bin/env python3
"""Benchmark YOLO-pose runtime on full frames vs tracked player ROIs.

This script is meant to answer one specific question:
does a larger YOLO-pose model or larger imgsz really cost more in this pipeline?

It measures wall-clock time with device synchronization around each prediction,
then writes a compact CSV so the result can be used in thesis notes.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default="practice_videos/Bryan_LR_Complete.mp4",
        help="Video to sample frames from.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"],
        help="YOLO-pose weight files to benchmark. Missing local files are skipped.",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO-pose benchmarks.",
    )
    parser.add_argument(
        "--include-movenet",
        action="store_true",
        help="Also benchmark MoveNet Lightning SinglePose and MoveNet Lightning MultiPose from TFHub.",
    )
    parser.add_argument(
        "--movenet-models",
        nargs="+",
        choices=["singlepose_lightning", "multipose_lightning"],
        default=["singlepose_lightning", "multipose_lightning"],
        help="MoveNet variants to benchmark when --include-movenet is set.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[192, 384],
        help="Image sizes to test.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for Ultralytics/PyTorch, e.g. auto, mps, cpu, cuda:0.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of measured frames after warmup.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup frames per benchmark case.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Read every Nth frame from the video.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting frame index.",
    )
    parser.add_argument(
        "--roi-tracks",
        nargs="+",
        default=["practice_videos/player1_track.npy", "practice_videos/player2_track.npy"],
        help="Track .npy files with rows [id, x1, y1, x2, y2].",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding in pixels around each tracked box before cropping.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Pose confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.35, help="Pose NMS IoU threshold.")
    parser.add_argument(
        "--output",
        default="benchmark_yolopose_runtime.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    if requested_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("WARNING: MPS was requested but is not available. Falling back to CPU.")
        return "cpu"

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"

    return requested_device


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def collect_frames(video_path: Path, total_frames: int, stride: int, start: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[tuple[int, np.ndarray]] = []

    while len(frames) < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if (frame_idx - start) % stride == 0:
            frames.append((frame_idx, frame))

    cap.release()
    if not frames:
        raise RuntimeError("No frames collected. Check --video, --start, and --stride.")

    return frames


def load_tracks(track_paths: list[str]) -> list[np.ndarray]:
    tracks = []
    for track_path in track_paths:
        path = Path(track_path)
        if not path.exists():
            print(f"Skipping missing track file: {path}")
            continue
        tracks.append(np.load(path))
    return tracks


def crop_track_roi(frame: np.ndarray, track_row: np.ndarray, padding: int) -> np.ndarray | None:
    if track_row.shape[0] >= 5:
        _, x1, y1, x2, y2 = track_row[:5]
    else:
        x1, y1, x2, y2 = track_row[:4]

    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return None

    height, width = frame.shape[:2]
    left = max(0, int(np.floor(x1 - padding)))
    top = max(0, int(np.floor(y1 - padding)))
    right = min(width, int(np.ceil(x2 + padding)))
    bottom = min(height, int(np.ceil(y2 + padding)))

    if right <= left or bottom <= top:
        return None

    return frame[top:bottom, left:right].copy()


def build_roi_pairs(
    frames: list[tuple[int, np.ndarray]],
    tracks: list[np.ndarray],
    padding: int,
) -> list[tuple[int, list[np.ndarray]]]:
    roi_pairs = []
    for frame_idx, frame in frames:
        rois = []
        valid = True
        for track in tracks:
            if frame_idx >= len(track):
                valid = False
                break
            roi = crop_track_roi(frame, track[frame_idx], padding)
            if roi is None:
                valid = False
                break
            rois.append(roi)

        if valid and rois:
            roi_pairs.append((frame_idx, rois))

    return roi_pairs


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.array(values, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.percentile(arr, 95)),
        float(np.std(arr)),
    )


def model_param_count(model: YOLO) -> int:
    return sum(param.numel() for param in model.model.parameters())


def format_csv_value(value: float | int | str) -> float | int | str:
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.2f}"
        return ""
    return value


def run_full_frame_benchmark(
    model: YOLO,
    frames: list[tuple[int, np.ndarray]],
    imgsz: int,
    device: str,
    warmup: int,
    conf: float,
    iou: float,
) -> dict[str, float | int | str]:
    wall_times = []
    ultra_infer_times = []

    for i, (_, frame) in enumerate(frames):
        sync_device(device)
        start = time.perf_counter()
        result = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        sync_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            wall_times.append(elapsed_ms)
            ultra_infer_times.append(float(result.speed.get("inference", np.nan)))

    mean_ms, median_ms, p95_ms, std_ms = summarize(wall_times)
    ultra_mean, _, _, _ = summarize(ultra_infer_times)
    return {
        "scope": "full_frame",
        "calls_per_frame": 1,
        "measured_frames": len(wall_times),
        "imgsz": imgsz,
        "wall_mean_ms": mean_ms,
        "wall_median_ms": median_ms,
        "wall_p95_ms": p95_ms,
        "wall_std_ms": std_ms,
        "ultralytics_infer_mean_ms": ultra_mean,
    }


def run_roi_pair_benchmark(
    model: YOLO,
    roi_pairs: list[tuple[int, list[np.ndarray]]],
    imgsz: int,
    device: str,
    warmup: int,
    conf: float,
    iou: float,
) -> dict[str, float | int | str]:
    wall_times = []
    ultra_infer_times = []

    for i, (_, rois) in enumerate(roi_pairs):
        sync_device(device)
        start = time.perf_counter()
        frame_infer_ms = 0.0
        for roi in rois:
            result = model(roi, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
            frame_infer_ms += float(result.speed.get("inference", np.nan))
        sync_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            wall_times.append(elapsed_ms)
            ultra_infer_times.append(frame_infer_ms)

    mean_ms, median_ms, p95_ms, std_ms = summarize(wall_times)
    ultra_mean, _, _, _ = summarize(ultra_infer_times)
    return {
        "scope": "roi_pair_total",
        "calls_per_frame": len(roi_pairs[0][1]) if roi_pairs else 0,
        "measured_frames": len(wall_times),
        "imgsz": imgsz,
        "wall_mean_ms": mean_ms,
        "wall_median_ms": median_ms,
        "wall_p95_ms": p95_ms,
        "wall_std_ms": std_ms,
        "ultralytics_infer_mean_ms": ultra_mean,
    }


class MoveNetRunner:
    def __init__(self, name: str, input_size: int, model_url: str) -> None:
        import tensorflow as tf
        import tensorflow_hub as hub

        self.name = name
        self.input_size = input_size
        self.tf = tf
        self.model = hub.load(model_url).signatures["serving_default"]

    def __call__(self, image_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self.tf.convert_to_tensor(frame_rgb)
        image = self.tf.expand_dims(image, axis=0)
        image = self.tf.image.resize_with_pad(image, self.input_size, self.input_size)
        image = self.tf.cast(image, dtype=self.tf.int32)
        outputs = self.model(image)

        # Force TensorFlow execution before stopping the timer.
        for value in outputs.values():
            _ = value.numpy()


def load_movenet_runner(model_name: str) -> MoveNetRunner:
    model_specs = {
        "singlepose_lightning": (
            192,
            "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        ),
        "multipose_lightning": (
            256,
            "https://tfhub.dev/google/movenet/multipose/lightning/1",
        ),
    }
    input_size, model_url = model_specs[model_name]
    return MoveNetRunner(model_name, input_size, model_url)


def run_movenet_full_frame_benchmark(
    runner: MoveNetRunner,
    frames: list[tuple[int, np.ndarray]],
    warmup: int,
) -> dict[str, float | int | str]:
    wall_times = []

    for i, (_, frame) in enumerate(frames):
        start = time.perf_counter()
        runner(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            wall_times.append(elapsed_ms)

    mean_ms, median_ms, p95_ms, std_ms = summarize(wall_times)
    return {
        "scope": "full_frame",
        "calls_per_frame": 1,
        "measured_frames": len(wall_times),
        "imgsz": runner.input_size,
        "wall_mean_ms": mean_ms,
        "wall_median_ms": median_ms,
        "wall_p95_ms": p95_ms,
        "wall_std_ms": std_ms,
        "ultralytics_infer_mean_ms": np.nan,
    }


def run_movenet_roi_pair_benchmark(
    runner: MoveNetRunner,
    roi_pairs: list[tuple[int, list[np.ndarray]]],
    warmup: int,
) -> dict[str, float | int | str]:
    wall_times = []

    for i, (_, rois) in enumerate(roi_pairs):
        start = time.perf_counter()
        for roi in rois:
            runner(roi)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if i >= warmup:
            wall_times.append(elapsed_ms)

    mean_ms, median_ms, p95_ms, std_ms = summarize(wall_times)
    return {
        "scope": "roi_pair_total",
        "calls_per_frame": len(roi_pairs[0][1]) if roi_pairs else 0,
        "measured_frames": len(wall_times),
        "imgsz": runner.input_size,
        "wall_mean_ms": mean_ms,
        "wall_median_ms": median_ms,
        "wall_p95_ms": p95_ms,
        "wall_std_ms": std_ms,
        "ultralytics_infer_mean_ms": np.nan,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    total_frames = args.frames + args.warmup
    video_path = Path(args.video)

    frames = collect_frames(video_path, total_frames, args.stride, args.start)
    tracks = load_tracks(args.roi_tracks)
    roi_pairs = build_roi_pairs(frames, tracks, args.padding) if tracks else []

    print(f"Video: {video_path}")
    print(f"Collected frames: {len(frames)}")
    print(f"ROI frame pairs: {len(roi_pairs)} from {len(tracks)} track file(s)")
    print(f"Requested device: {args.device}")
    print(f"Actual device: {device}")

    rows: list[dict[str, float | int | str]] = []
    if not args.skip_yolo:
        for weight_name in args.weights:
            weight_path = Path(weight_name)
            if not weight_path.exists():
                print(f"Skipping missing weights: {weight_path}")
                continue

            print(f"\nLoading {weight_path} ...")
            model = YOLO(str(weight_path))
            model.to(device)
            params = model_param_count(model)
            print(f"Parameters: {params:,}")

            for imgsz in args.imgsz:
                print(f"Benchmarking {weight_path.name}, imgsz={imgsz}, full frame")
                full_row = run_full_frame_benchmark(
                    model, frames, imgsz, device, args.warmup, args.conf, args.iou
                )
                full_row.update(
                    {"backend": "ultralytics", "weights": weight_path.name, "params": params, "device": device}
                )
                rows.append(full_row)

                if roi_pairs:
                    print(f"Benchmarking {weight_path.name}, imgsz={imgsz}, ROI pair")
                    roi_row = run_roi_pair_benchmark(
                        model, roi_pairs, imgsz, device, args.warmup, args.conf, args.iou
                    )
                    roi_row.update(
                        {"backend": "ultralytics", "weights": weight_path.name, "params": params, "device": device}
                    )
                    rows.append(roi_row)

    if args.include_movenet:
        for model_name in args.movenet_models:
            print(f"\nLoading MoveNet {model_name} ...")
            try:
                runner = load_movenet_runner(model_name)
            except Exception as exc:
                print(f"Skipping MoveNet {model_name}: {exc}")
                continue

            print(f"Benchmarking MoveNet {model_name}, input={runner.input_size}, full frame")
            full_row = run_movenet_full_frame_benchmark(runner, frames, args.warmup)
            full_row.update(
                {"backend": "tensorflow_hub", "weights": f"movenet_{model_name}", "params": "", "device": "tensorflow"}
            )
            rows.append(full_row)

            if roi_pairs:
                print(f"Benchmarking MoveNet {model_name}, input={runner.input_size}, ROI pair")
                roi_row = run_movenet_roi_pair_benchmark(runner, roi_pairs, args.warmup)
                roi_row.update(
                    {
                        "backend": "tensorflow_hub",
                        "weights": f"movenet_{model_name}",
                        "params": "",
                        "device": "tensorflow",
                    }
                )
                rows.append(roi_row)

    if not rows:
        raise RuntimeError("No benchmark rows were produced. Check weight paths.")

    output_path = Path(args.output)
    fieldnames = [
        "backend",
        "weights",
        "params",
        "device",
        "scope",
        "calls_per_frame",
        "measured_frames",
        "imgsz",
        "wall_mean_ms",
        "wall_median_ms",
        "wall_p95_ms",
        "wall_std_ms",
        "ultralytics_infer_mean_ms",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_csv_value(row.get(field, "")) for field in fieldnames})

    print(f"\nSaved: {output_path}")
    print("\nSummary:")
    for row in rows:
        print(
            f"{row['weights']:28s} imgsz={row['imgsz']:>3} {row['scope']:15s} "
            f"wall median={row['wall_median_ms']:6.2f} ms "
            f"p95={row['wall_p95_ms']:6.2f} ms"
        )


if __name__ == "__main__":
    main()
