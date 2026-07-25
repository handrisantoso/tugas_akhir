#!/usr/bin/env python3
import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Analyze RAGAS metrics by search mode.")
    parser.add_argument(
        "results_csv", 
        type=str, 
        nargs="?",
        default="eval/ragas_eval/results/jina_ragas_2/per_query.csv",
        help="Path to the per_query.csv file from RAGAS evaluation."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="eval/eval_dataset.csv",
        help="Path to the original eval_dataset.csv"
    )
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    dataset_path = Path(args.dataset)

    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return 1
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return 1

    # Load search modes from dataset
    query_modes = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_modes[row["query_id"]] = row["expected_search_mode"]

    # Load metrics from results
    mode_metrics = defaultdict(lambda: {"faithfulness": [], "answer_relevancy": []})
    
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row["query_id"]
            if q_id not in query_modes:
                continue
            
            mode = query_modes[q_id]
            
            try:
                f_score = float(row["faithfulness"]) if row["faithfulness"] else None
                r_score = float(row["answer_relevancy"]) if row["answer_relevancy"] else None
            except ValueError:
                continue

            if f_score is not None:
                mode_metrics[mode]["faithfulness"].append(f_score)
            if r_score is not None:
                mode_metrics[mode]["answer_relevancy"].append(r_score)

    # Print table
    print(f"\nMetric Results by Search Mode for: {results_path.parent.name}\n")
    print(f"{'Search Mode':<15} | {'Count':<10} | {'Faithfulness':<15} | {'Answer Relevancy'}")
    print("-" * 65)

    # Sort modes (TEXT, IMAGE, HYBRID)
    for mode in sorted(mode_metrics.keys(), key=lambda m: (m != "TEXT", m != "IMAGE", m)):
        f_scores = mode_metrics[mode]["faithfulness"]
        r_scores = mode_metrics[mode]["answer_relevancy"]
        
        count = len(f_scores) # Assuming both lists are similar length
        avg_f = sum(f_scores) / len(f_scores) if f_scores else 0.0
        avg_r = sum(r_scores) / len(r_scores) if r_scores else 0.0

        print(f"{mode:<15} | {count:<10} | {avg_f:<15.4f} | {avg_r:.4f}")
    print()

if __name__ == "__main__":
    sys.exit(main())
