"""Optional check against original, unstripped upstream effective-pathway JSON output.

Usage: python3 pathway_ablation/check_vs_gate2.py <ablation_output_dir> [original_jsons_dir]
"""
import json
import sys
from pathlib import Path

ablation = Path(sys.argv[1])
default_ref = "/workspace/outputs/reproduction/tvbench/effective-pathway-7b/LLaVA-NeXT-7B-Video-FT/jsons"
gate = Path(sys.argv[2] if len(sys.argv) > 2 else default_ref)

checked = bad = 0
for task_dir in sorted(p for p in ablation.iterdir() if p.is_dir()):
    mine = task_dir / "effective.jsonl"
    ref = gate / f"{task_dir.name}.json"
    if not mine.exists() or not ref.exists():
        print("skip", task_dir.name, "(missing file)")
        continue
    with ref.open() as source:
        ref_by_id = {r["data_id"]: r for r in json.load(source)}
    with mine.open() as source:
        for line in source:
            if not line.strip():
                continue
            r = json.loads(line)
            g = ref_by_id.get(r["data_id"])
            if g is None:
                print("not in reproduction:", task_dir.name, r["data_id"])
                bad += 1
                continue
            checked += 1
            same_answers = r["base_answer"] == g["base_answer"] and r["new_answer"] == g["new_answer"]
            d_base = abs(r["base_score"] - g["base_score"])
            d_new = abs(r["new_score"] - g["new_score"])
            same_edges = r["new_attention_cnt"] == g["new_attention_cnt"]
            if not (same_answers and d_base < 1e-3 and d_new < 1e-3 and same_edges):
                bad += 1
                print("MISMATCH", task_dir.name, r["data_id"],
                      "answers", (r["base_answer"], r["new_answer"]), (g["base_answer"], g["new_answer"]),
                      "dbase=%.5f dnew=%.5f" % (d_base, d_new), "edges", r["new_attention_cnt"], g["new_attention_cnt"])

print(f"checked={checked} mismatches={bad}")
print("PASS" if checked and not bad else "CHECK")
