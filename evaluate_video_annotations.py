"""Evaluate end-to-end Tekken video predictions against saved annotations.

The evaluator uses the exact 30-frame windows stored in the MMAction dataset
split. This avoids scoring training windows when ``--split val`` is selected
and keeps evaluation aligned with the temporal windows used by ST-GCN++.

Run ``tekken_video_inference.py`` with ``--predictions-json`` first, then pass
that JSON file to this script.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "practice_videos"
WINDOW_RANGE_RE = re.compile(r"_(?:w|f)(\d+)_(\d+)(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score logged end-to-end video predictions against the annotated "
            "MMAction train/validation windows."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Prediction JSON written by tekken_video_inference.py.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DATA_DIR / "Bryan_LR_Complete.json",
        help="Original frame-range annotation JSON.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR / "skeleton_dataset_all_moves_idle.pkl",
        help="MMAction PoseDataset pickle containing split membership and windows.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DATA_DIR / "move_labels_all_moves_idle.json",
        help="Label-to-index JSON used by training and inference.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="val",
        help="Dataset windows to evaluate. Use val for the defensible internal result.",
    )
    parser.add_argument(
        "--nearest-frame-tolerance",
        type=int,
        default=2,
        help=(
            "Maximum distance between a dataset window endpoint and a logged "
            "prediction frame. Two frames covers predict_every=3."
        ),
    )
    parser.add_argument(
        "--event-pre-tolerance",
        type=int,
        default=5,
        help="Frames before an annotated event considered for detection timing.",
    )
    parser.add_argument(
        "--event-post-tolerance",
        type=int,
        default=20,
        help="Frames after an annotated event considered for rolling-window detection.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Evaluation JSON path. Defaults beside the prediction JSON.",
    )
    parser.add_argument(
        "--confusion-csv",
        type=Path,
        default=None,
        help="Confusion-matrix CSV path. Defaults beside the prediction JSON.",
    )
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def harmonic_f1(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def player_from_name(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith(("p1_", "player1_")):
        return "player1"
    if lowered.startswith(("p2_", "player2_")):
        return "player2"
    raise ValueError(f"Could not infer player from dataset item: {name}")


def window_range(frame_dir: str) -> tuple[int, int]:
    match = WINDOW_RANGE_RE.search(frame_dir)
    if match is None:
        raise ValueError(
            f"Dataset item does not encode a frame range with _w... or _f...: {frame_dir}"
        )
    return int(match.group(1)), int(match.group(2))


def build_prediction_index(
    records: list[dict[str, Any]],
) -> dict[str, tuple[list[int], list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        player = str(record.get("player", "")).lower()
        if player not in {"player1", "player2"}:
            continue
        grouped[player].append(record)

    indexed = {}
    for player, player_records in grouped.items():
        player_records.sort(key=lambda item: int(item["frame"]))
        indexed[player] = (
            [int(item["frame"]) for item in player_records],
            player_records,
        )
    return indexed


def nearest_prediction(
    index: dict[str, tuple[list[int], list[dict[str, Any]]]],
    player: str,
    target_frame: int,
    tolerance: int,
) -> dict[str, Any] | None:
    if player not in index:
        return None
    frames, records = index[player]
    insertion = bisect.bisect_left(frames, target_frame)
    candidates = []
    for candidate_index in (insertion - 1, insertion):
        if 0 <= candidate_index < len(records):
            candidates.append(records[candidate_index])
    if not candidates:
        return None
    closest = min(
        candidates,
        key=lambda item: (
            abs(int(item["frame"]) - target_frame),
            int(item["frame"]),
        ),
    )
    if abs(int(closest["frame"]) - target_frame) > tolerance:
        return None
    return closest


def top_indices(probabilities: list[float], count: int) -> list[int]:
    return sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )[:count]


def classification_metrics(
    rows: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    class_count = len(labels)
    confusion = [[0 for _ in labels] for _ in labels]
    losses = []
    top5_hits = 0
    stable_correct = 0
    accepted = 0
    accepted_correct = 0

    for row in rows:
        truth = int(row["truth_index"])
        predicted = int(row["predicted_index"])
        probabilities = [float(value) for value in row["probabilities"]]
        confusion[truth][predicted] += 1
        losses.append(-math.log(max(probabilities[truth], 1e-15)))
        top5_hits += int(truth in top_indices(probabilities, min(5, class_count)))
        stable_correct += int(row.get("stable_label") == labels[truth])
        if row.get("thresholded_label") != "uncertain":
            accepted += 1
            accepted_correct += int(predicted == truth)

    total = len(rows)
    per_class = []
    macro_precision_values = []
    macro_recall_values = []
    macro_f1_values = []
    weighted_f1_total = 0.0

    for class_index, label in enumerate(labels):
        true_positive = confusion[class_index][class_index]
        support = sum(confusion[class_index])
        predicted_count = sum(row[class_index] for row in confusion)
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        f1 = harmonic_f1(precision, recall)
        if support:
            macro_precision_values.append(precision)
            macro_recall_values.append(recall)
            macro_f1_values.append(f1)
            weighted_f1_total += f1 * support
        per_class.append(
            {
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    correct = sum(confusion[index][index] for index in range(class_count))
    return {
        "samples": total,
        "top1_accuracy": safe_div(correct, total),
        "top5_accuracy": safe_div(top5_hits, total),
        "cross_entropy_loss": mean(losses) if losses else 0.0,
        "macro_precision": mean(macro_precision_values) if macro_precision_values else 0.0,
        "macro_recall": mean(macro_recall_values) if macro_recall_values else 0.0,
        "macro_f1": mean(macro_f1_values) if macro_f1_values else 0.0,
        "weighted_f1": safe_div(weighted_f1_total, total),
        "accepted_prediction_coverage": safe_div(accepted, total),
        "accepted_prediction_accuracy": safe_div(accepted_correct, accepted),
        "stable_display_accuracy": safe_div(stable_correct, total),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "rows_are_ground_truth": True,
            "matrix": confusion,
        },
    }


def aggregate_event_rows(
    window_rows: list[dict[str, Any]],
    labels: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        if row["truth_label"] == "idle":
            continue
        grouped[str(row["source_frame_dir"])].append(row)

    event_rows = []
    for source_frame_dir, rows in sorted(grouped.items()):
        probability_count = len(rows[0]["probabilities"])
        averaged_probabilities = [
            mean(float(row["probabilities"][index]) for row in rows)
            for index in range(probability_count)
        ]
        predicted_index = max(
            range(probability_count),
            key=lambda index: averaged_probabilities[index],
        )
        truth_index = int(rows[0]["truth_index"])
        event_rows.append(
            {
                "source_frame_dir": source_frame_dir,
                "player": rows[0]["player"],
                "truth_index": truth_index,
                "truth_label": labels[truth_index],
                "predicted_index": predicted_index,
                "predicted_label": labels[predicted_index],
                "probabilities": averaged_probabilities,
                "thresholded_label": labels[predicted_index],
                "stable_label": max(
                    (str(row.get("stable_label", "")) for row in rows),
                    key=lambda label: sum(
                        str(candidate.get("stable_label", "")) == label
                        for candidate in rows
                    ),
                ),
                "window_count": len(rows),
            }
        )
    return event_rows


def event_detection_metrics(
    source_events: set[str],
    annotations: dict[str, dict[str, Any]],
    prediction_index: dict[str, tuple[list[int], list[dict[str, Any]]]],
    pre_tolerance: int,
    post_tolerance: int,
    label_field: str,
) -> dict[str, Any]:
    detections = []
    missed = []

    for source_event in sorted(source_events):
        annotation = annotations[source_event]
        player = str(annotation["player"])
        truth_label = f"{annotation['character']} {annotation['move']}"
        start = int(annotation["start_frame"]) - pre_tolerance
        end = int(annotation["end_frame"]) + post_tolerance
        player_records = prediction_index.get(player, ([], []))[1]
        candidates = [
            record
            for record in player_records
            if start <= int(record["frame"]) <= end
            and record.get(label_field) == truth_label
        ]
        if candidates:
            detection_frame = min(int(record["frame"]) for record in candidates)
            detections.append(
                {
                    "source_frame_dir": source_event,
                    "truth_label": truth_label,
                    "player": player,
                    "detection_frame": detection_frame,
                    "delay_from_annotation_end_frames": (
                        detection_frame - int(annotation["end_frame"])
                    ),
                }
            )
        else:
            missed.append(
                {
                    "source_frame_dir": source_event,
                    "truth_label": truth_label,
                    "player": player,
                }
            )

    delays = [
        int(item["delay_from_annotation_end_frames"]) for item in detections
    ]
    return {
        "events": len(source_events),
        "detected_events": len(detections),
        "event_recall": safe_div(len(detections), len(source_events)),
        "mean_delay_from_annotation_end_frames": mean(delays) if delays else None,
        "median_delay_from_annotation_end_frames": median(delays) if delays else None,
        "pre_tolerance_frames": pre_tolerance,
        "post_tolerance_frames": post_tolerance,
        "prediction_label_field": label_field,
        "detections": detections,
        "missed": missed,
        "note": (
            "This is detection recall and timing inside annotated-event windows. "
            "Full-video event precision requires explicit background/ignore annotations."
        ),
    }


def write_confusion_csv(path: Path, metrics: dict[str, Any]) -> None:
    confusion = metrics["confusion_matrix"]
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["ground_truth/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.predictions, "Prediction JSON"),
        (args.annotations, "Annotation JSON"),
        (args.dataset, "Dataset pickle"),
        (args.labels, "Label map"),
    ):
        require_path(path, label)

    prediction_report = load_json(args.predictions)
    annotations = load_json(args.annotations)
    dataset = load_pickle(args.dataset)
    label_to_index = load_json(args.labels)
    labels = [
        label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])
    ]

    prediction_labels = prediction_report.get("labels")
    if prediction_labels != labels:
        raise ValueError(
            "Prediction-label order does not match the training label map. "
            f"Predictions={prediction_labels}, expected={labels}"
        )

    records = prediction_report.get("predictions", [])
    if not records:
        raise ValueError("Prediction JSON contains no prediction records.")
    prediction_index = build_prediction_index(records)

    annotations_by_name = {
        str(item["frame_dir"]): item for item in dataset["annotations"]
    }
    if args.split == "all":
        selected_names = list(annotations_by_name)
    else:
        selected_names = list(dataset["split"][args.split])

    matched_rows = []
    unmatched_windows = []
    selected_move_sources: set[str] = set()

    for frame_dir in selected_names:
        dataset_annotation = annotations_by_name[frame_dir]
        source_frame_dir = str(
            dataset_annotation.get("source_frame_dir", frame_dir)
        )
        player = player_from_name(source_frame_dir)
        start_frame, end_frame = window_range(frame_dir)
        truth_index = int(dataset_annotation["label"])
        if labels[truth_index] != "idle":
            selected_move_sources.add(source_frame_dir)
        prediction = nearest_prediction(
            prediction_index,
            player,
            end_frame,
            args.nearest_frame_tolerance,
        )
        if prediction is None:
            unmatched_windows.append(
                {
                    "frame_dir": frame_dir,
                    "source_frame_dir": source_frame_dir,
                    "player": player,
                    "expected_prediction_frame": end_frame,
                    "truth_label": labels[truth_index],
                }
            )
            continue

        probabilities = [
            float(value) for value in prediction["probabilities"]
        ]
        if len(probabilities) != len(labels):
            raise ValueError(
                f"Prediction at frame {prediction['frame']} has "
                f"{len(probabilities)} probabilities; expected {len(labels)}."
            )
        predicted_index = max(
            range(len(probabilities)),
            key=lambda index: probabilities[index],
        )
        matched_rows.append(
            {
                "frame_dir": frame_dir,
                "source_frame_dir": source_frame_dir,
                "player": player,
                "window_start_frame": start_frame,
                "window_end_frame": end_frame,
                "prediction_frame": int(prediction["frame"]),
                "prediction_frame_delta": int(prediction["frame"]) - end_frame,
                "truth_index": truth_index,
                "truth_label": labels[truth_index],
                "predicted_index": predicted_index,
                "predicted_label": labels[predicted_index],
                "probabilities": probabilities,
                "thresholded_label": prediction.get("thresholded_label"),
                "stable_label": prediction.get("stable_label"),
            }
        )

    if not matched_rows:
        raise RuntimeError(
            "No dataset windows matched prediction frames. Check that the "
            "prediction run covers the full video and uses the same frame numbering."
        )

    window_metrics = classification_metrics(matched_rows, labels)
    window_metrics["prediction_coverage"] = safe_div(
        len(matched_rows),
        len(selected_names),
    )
    window_metrics["top1_accuracy_including_missing"] = (
        window_metrics["top1_accuracy"] * window_metrics["prediction_coverage"]
    )
    window_metrics["top5_accuracy_including_missing"] = (
        window_metrics["top5_accuracy"] * window_metrics["prediction_coverage"]
    )
    event_rows = aggregate_event_rows(matched_rows, labels)
    event_metrics = classification_metrics(event_rows, labels)
    event_metrics["event_window_coverage"] = safe_div(
        len(event_rows),
        len(selected_move_sources),
    )
    event_metrics["top1_accuracy_including_unmatched_events"] = (
        event_metrics["top1_accuracy"] * event_metrics["event_window_coverage"]
    )
    detection_metrics = event_detection_metrics(
        selected_move_sources,
        annotations,
        prediction_index,
        args.event_pre_tolerance,
        args.event_post_tolerance,
        "thresholded_label",
    )
    stable_detection_metrics = event_detection_metrics(
        selected_move_sources,
        annotations,
        prediction_index,
        args.event_pre_tolerance,
        args.event_post_tolerance,
        "stable_label",
    )

    idle_rows = [row for row in matched_rows if row["truth_label"] == "idle"]
    accepted_idle_false_positives = sum(
        row.get("thresholded_label") not in {"idle", "uncertain"}
        for row in idle_rows
    )
    idle_metrics = {
        "explicit_idle_windows": len(idle_rows),
        "accepted_non_idle_false_positives": accepted_idle_false_positives,
        "accepted_non_idle_false_positive_rate": safe_div(
            accepted_idle_false_positives,
            len(idle_rows),
        ),
        "note": (
            "Only explicitly sampled validation idle windows are scored. "
            "Unannotated video gaps are not automatically treated as idle."
        ),
    }

    output_json = args.output_json or args.predictions.with_name(
        f"{args.predictions.stem}_{args.split}_evaluation.json"
    )
    confusion_csv = args.confusion_csv or args.predictions.with_name(
        f"{args.predictions.stem}_{args.split}_confusion.csv"
    )
    output = {
        "evaluation_scope": (
            "Internal end-to-end evaluation on exact MMAction dataset windows. "
            "This is not an unseen-recording generalization result."
        ),
        "split": args.split,
        "prediction_file": str(args.predictions),
        "annotation_file": str(args.annotations),
        "dataset_file": str(args.dataset),
        "checkpoint": prediction_report.get("checkpoint"),
        "predict_every": prediction_report.get("predict_every"),
        "window_size": prediction_report.get("window_size"),
        "selected_windows": len(selected_names),
        "matched_windows": len(matched_rows),
        "unmatched_windows": len(unmatched_windows),
        "unique_move_events": len(selected_move_sources),
        "window_metrics": window_metrics,
        "event_metrics_mean_probability": event_metrics,
        "event_detection": detection_metrics,
        "stable_event_detection": stable_detection_metrics,
        "idle_metrics": idle_metrics,
        "unmatched_window_details": unmatched_windows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file_obj:
        json.dump(output, file_obj, indent=2)
    write_confusion_csv(confusion_csv, window_metrics)

    print(f"Evaluation split: {args.split}")
    print(
        f"Matched windows: {len(matched_rows)}/{len(selected_names)} "
        f"(unmatched {len(unmatched_windows)})"
    )
    print(
        "Window metrics: "
        f"Top-1={window_metrics['top1_accuracy']:.4f}, "
        f"Top-5={window_metrics['top5_accuracy']:.4f}, "
        f"macro-F1={window_metrics['macro_f1']:.4f}, "
        f"loss={window_metrics['cross_entropy_loss']:.4f}, "
        f"coverage={window_metrics['prediction_coverage']:.4f}, "
        "strict Top-1 including missing="
        f"{window_metrics['top1_accuracy_including_missing']:.4f}"
    )
    print(
        "Event metrics (mean probability across offsets): "
        f"Top-1={event_metrics['top1_accuracy']:.4f}, "
        f"macro-F1={event_metrics['macro_f1']:.4f}"
    )
    print(
        "Annotated-event detection (thresholded raw label): "
        f"{detection_metrics['detected_events']}/"
        f"{detection_metrics['events']} "
        f"({detection_metrics['event_recall']:.4f})"
    )
    print(
        "Annotated-event detection (stable displayed label): "
        f"{stable_detection_metrics['detected_events']}/"
        f"{stable_detection_metrics['events']} "
        f"({stable_detection_metrics['event_recall']:.4f})"
    )
    print(f"Saved evaluation: {output_json}")
    print(f"Saved confusion matrix: {confusion_csv}")


if __name__ == "__main__":
    main()
