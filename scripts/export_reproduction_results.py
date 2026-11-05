"""Export the selected upstream TVBench fields as question-level JSONL."""

import argparse
import json
from pathlib import Path


FIELDS = (
    "condition", "task_type", "data_id", "gt", "base_answer", "new_answer",
    "base_score", "new_score", "relative_diff", "base_score_gt", "new_score_gt",
    "base_attention_cnt", "new_attention_cnt", "attention_amount",
)
SOURCE_FIELDS = {"base_score_gt": "gt_score"}
TARGETS = {"effective-pathway-7b": "effective", "random": "random"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Upstream reproduction/tvbench directory")
    parser.add_argument("destination", type=Path, help="Public results/reproduction directory")
    args = parser.parse_args()

    for target, condition in TARGETS.items():
        source_dir = args.source / target / "LLaVA-NeXT-7B-Video-FT" / "jsons"
        paths = sorted(source_dir.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No JSON files in {source_dir}")
        for path in paths:
            if "_" not in path.stem:
                raise ValueError(f"Expected numbered task name: {path.name}")
            task_type = path.stem.split("_", 1)[1]
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Expected a list in {path}")
            destination_dir = args.destination / path.stem
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{condition}.jsonl"
            with destination.open("w", encoding="utf-8") as output:
                for source in rows:
                    record = {
                        field: (
                            condition if field == "condition" else
                            task_type if field == "task_type" else
                            source[SOURCE_FIELDS.get(field, field)]
                        )
                        for field in FIELDS
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{destination}: {len(rows)} rows")


if __name__ == "__main__":
    main()
