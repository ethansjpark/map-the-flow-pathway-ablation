"""Summarize pathway ablation JSONL files by task and across tasks."""

import json
from pathlib import Path
import sys

from pathway_conditions import CONDITIONS


def read_task(task_dir):
    by_condition = {}
    baseline_by_id = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        rows = {}
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    record = json.loads(line)
                    rows[record["data_id"]] = record
                    baseline_by_id.setdefault(record["data_id"], record)
        by_condition[path.stem] = list(rows.values())

    if baseline_by_id:
        by_condition["full"] = [
            {
                **record,
                "condition": "full",
                "new_answer": record["base_answer"],
                "new_score": record["base_score"],
                "new_score_gt": record["base_score_gt"],
                "relative_diff": 0.0,
                "new_attention_cnt": record["base_attention_cnt"],
            }
            for record in baseline_by_id.values()
        ]
    return by_condition


def percent(numerator, denominator):
    return f"{100 * numerator / denominator:.2f}" if denominator else "N/A"


def print_row(task, condition, rows):
    correct = sum(row["new_answer"] == row["gt"] for row in rows)
    baseline_correct = [
        row for row in rows if row["base_answer"].lower() == row["gt"].lower()
    ]
    correct_after = sum(row["new_answer"] == row["gt"] for row in baseline_correct)
    mean_diff = (
        f"{sum(row['relative_diff'] for row in baseline_correct) / len(baseline_correct):.2f}"
        if baseline_correct else "N/A"
    )
    base_edges = sum(row["base_attention_cnt"] for row in rows)
    kept_edges = sum(row["new_attention_cnt"] for row in rows)
    print(
        f"{task:<23} {condition:<17} {len(rows):>5} "
        f"{percent(correct, len(rows)):>9} "
        f"{percent(correct_after, len(baseline_correct)):>17} "
        f"{mean_diff:>12} {percent(kept_edges, base_edges):>11}"
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python pathway_ablation/summarize_ablation.py <output_dir>")
    root = Path(sys.argv[1])
    pooled = {}
    print(f"{'task':<23} {'condition':<17} {'n':>5} {'accuracy%':>9} "
          f"{'base-correct acc%':>17} {'mean Δp%':>12} {'edges kept%':>11}")
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        by_condition = read_task(task_dir)
        for condition in CONDITIONS:
            rows = by_condition.get(condition, [])
            if not rows:
                continue
            print_row(task_dir.name, condition, rows)
            pooled.setdefault(condition, []).extend(rows)
    for condition in CONDITIONS:
        rows = pooled.get(condition, [])
        if rows:
            print_row("POOLED", condition, rows)


if __name__ == "__main__":
    main()
