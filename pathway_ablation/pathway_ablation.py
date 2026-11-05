# Adapted from analysis/effective_pathway_analysis.py of https://github.com/byminji/map-the-flow (commit 566f3ed064d324408fdf96f7a921ebf1e65f0908). Not affiliated with the original authors.
"""Run one baseline and several pathway knockouts per TVBench sample."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys

from pathway_conditions import CONDITIONS, build_block_layer_lists


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default="/workspace/map-the-flow")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pooling_shape", default="8-12-12")
    parser.add_argument("--conv_mode", default="eval_mvbench")
    parser.add_argument("--dataset_name", default="tvbench")
    parser.add_argument("--task_ids", default="3,6,8")
    parser.add_argument("--n_per_task", type=int, default=150)
    parser.add_argument("--conditions", default=",".join(name for name in CONDITIONS if name != "full"))
    parser.add_argument("--random_block_ratio", type=float, default=0.58)
    return parser.parse_args()


def completed_ids(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as source:
        return {json.loads(line)["data_id"] for line in source if line.strip()}


def append_result(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main():
    args = parse_args()
    sys.path.insert(0, args.repo_root)

    import torch
    from tqdm import tqdm
    from tasks.eval.eval_utils import conv_templates
    from tasks.eval.model_utils import load_model_and_dataset
    from analysis.causal_intervention_tools import (
        decode_tokens, find_inter_frame_block_ranges, find_token_range,
        precompute_attention_masks, precompute_random_block_attention_masks,
        predict_from_input, trace_with_attn_block,
    )

    torch.set_grad_enabled(False)
    if args.dataset_name in {"tvbench_open_ended", "videomme_open_ended", "vcgbench"}:
        raise NotImplementedError("Open-ended datasets are not supported")
    task_ids = {int(value) for value in args.task_ids.split(",")}
    conditions = list(dict.fromkeys(args.conditions.split(",")))
    if not conditions or any(name not in CONDITIONS for name in conditions):
        raise ValueError(f"Conditions must be from {list(CONDITIONS)}")
    if args.n_per_task < 1 or not 0 < args.random_block_ratio <= 1:
        raise ValueError("n_per_task must be positive and random_block_ratio must be in (0, 1]")

    pooling_shape = tuple(map(int, args.pooling_shape.split("-")))
    model, processor, dataset = load_model_and_dataset(
        rank=0, world_size=1,
        pretrained_model_name_or_path=args.model_path,
        num_frames=pooling_shape[0], use_lora=False,
        weight_dir=args.model_path, dataset_name=args.dataset_name,
        lora_alpha=0, lora_target_modules=["q_proj", "v_proj"],
        pooling_shape=pooling_shape, force_eager=True,
    )
    if "llava" not in args.model_path.lower():
        raise NotImplementedError("This runner supports LLaVA-NeXT-7B-Video-FT only")
    num_layers = model.config.text_config.num_hidden_layers
    if num_layers != 32:
        raise ValueError(f"Expected 32 layers, found {num_layers}")
    mask_num_heads = model.config.text_config.num_attention_heads
    image_token_index = model.config.image_token_index

    post_query_prompt = "\nOnly give the best option."
    answer_prompt = "Best option:" if args.dataset_name == "videomme" else "Best option:("
    video_index_map = defaultdict(list)
    for idx, entry in enumerate(dataset.data_list):
        video_index_map[entry["task_type"]].append(idx)

    for task_i, (task_type, qa_indices) in enumerate(sorted(video_index_map.items())):
        if task_i not in task_ids:
            continue
        random.seed(42)
        qa_indices = random.sample(qa_indices, min(args.n_per_task, len(qa_indices)))
        task_dir = Path(args.output_dir) / f"{task_i:02d}_{task_type}"
        task_dir.mkdir(parents=True, exist_ok=True)
        paths = {name: task_dir / f"{name}.jsonl" for name in conditions}
        seen = {name: completed_ids(path) for name, path in paths.items()}

        for sample_index, data_idx in enumerate(tqdm(qa_indices, desc=f"{task_i:02d} {task_type}")):
            pending = [name for name in conditions if data_idx not in seen[name]]
            if not pending:
                continue
            example = dataset[data_idx]
            video_list = example["video_pils"]
            conv = conv_templates[args.conv_mode].copy()
            conv.user_query(example["question"], None, post_query_prompt, is_mm=True)
            conv.assistant_response(answer_prompt)

            torch.cuda.empty_cache()
            prompt = conv.get_prompt()
            inputs = processor(text=prompt, images=video_list, return_tensors="pt").to(model.device)
            inputs["media_type"] = "video"
            input_ids = inputs["input_ids"][0]
            image_placeholder_index = torch.where(input_ids == image_token_index)[0].item()
            num_vis = pooling_shape[0] * pooling_shape[1] * pooling_shape[2]
            vision_range = [x + image_placeholder_index for x in range(num_vis)]
            ntoks = inputs["input_ids"].shape[1] + num_vis - 1
            last_token = ntoks - 1
            question_range = find_token_range(processor.tokenizer, input_ids, example["question"])
            assert question_range[0] > -1
            question_range = [x + num_vis - 1 for x in range(question_range[0], question_range[1])]

            answer_t, base_score, probs = [part[0] for part in predict_from_input(model, inputs)]
            base_score = base_score.cpu().item()
            [base_answer] = decode_tokens(processor.tokenizer, [answer_t])
            gt = example["answer"][1]
            gt_t = processor.tokenizer.get_vocab()[gt]
            base_score_gt = probs[gt_t].cpu().item()
            torch.cuda.empty_cache()

            query_lists, key_lists = find_inter_frame_block_ranges(
                vision_range, num_frames=pooling_shape[0],
                num_vis_one_frame=pooling_shape[1] * pooling_shape[2],
                vis_start_id=image_placeholder_index,
            )
            baseline_edges = num_layers * (ntoks * (ntoks + 1) // 2)

            for condition in pending:
                if condition == "full":
                    new_score, new_score_gt, new_answer = base_score, base_score_gt, base_answer
                    base_edges = new_edges = baseline_edges
                else:
                    if condition == "random":
                        torch.manual_seed(42 + sample_index)
                        num_block = int((ntoks * (ntoks + 1) // 2) * args.random_block_ratio)
                        rows, cols = torch.tril_indices(ntoks, ntoks, offset=0, device=model.device)
                        valid = (rows >= vision_range[0]) & (cols >= vision_range[0])
                        valid_indices = torch.stack([rows[valid], cols[valid]], dim=1)
                    else:
                        layers = build_block_layer_lists(CONDITIONS[condition], num_layers)
                        block_mappings = {
                            "vision_inter_frame": (key_lists, query_lists, layers["vision_inter_frame"]),
                            "vision_to_question": ([vision_range], [question_range], layers["vision_to_question"]),
                            "question_to_last": ([question_range], [[last_token]], layers["question_to_last"]),
                            "last_to_last": ([[last_token]], [[last_token]], layers["last_to_last"]),
                            "vision_to_last": ([vision_range], [[last_token]], layers["vision_to_last"]),
                            "vision_to_vision": ([vision_range], [vision_range], layers["vision_to_vision"]),
                            "question_to_question": ([question_range], [question_range], layers["question_to_question"]),
                        }

                    attn_mask_all_layers = []
                    base_edges, new_edges = 0, 0
                    for layer in range(num_layers):
                        if condition == "random":
                            attn_mask = precompute_random_block_attention_masks(
                                ntoks, mask_num_heads, valid_indices=valid_indices, N=num_block,
                                dtype=model.dtype, device=model.device,
                            )
                        else:
                            block_key_range, block_query_range = [], []
                            for keys, queries, blocked_layers in block_mappings.values():
                                if layer not in blocked_layers:
                                    continue
                                block_key_range.extend(keys)
                                block_query_range.extend(queries)
                            attn_mask = precompute_attention_masks(
                                ntoks, mask_num_heads, block_query_range, block_key_range,
                                model.dtype, model.device,
                            )
                        seq_len = attn_mask.size(-1)
                        base_edges += seq_len * (seq_len + 1) // 2
                        new_edges += (attn_mask[0, 0] == 0).sum().item()
                        attn_mask_all_layers.append(attn_mask)

                    r_answer, r_gt, _, new_answer_t = trace_with_attn_block(
                        model, inputs, answer_t, gt_t, attn_mask_all_layers, None, "llava",
                    )
                    new_score = r_answer.cpu().item()
                    new_score_gt = r_gt.cpu().item()
                    [new_answer] = decode_tokens(processor.tokenizer, [new_answer_t.cpu().item()])
                    del attn_mask_all_layers

                record = {
                    "condition": condition, "task_type": task_type, "data_id": data_idx,
                    "gt": gt, "base_answer": base_answer, "new_answer": new_answer,
                    "base_score": base_score, "new_score": new_score,
                    "relative_diff": (new_score - base_score) * 100.0 / base_score,
                    "base_score_gt": base_score_gt, "new_score_gt": new_score_gt,
                    "base_attention_cnt": base_edges, "new_attention_cnt": new_edges,
                    "attention_amount": new_edges / base_edges * 100,
                }
                append_result(paths[condition], record)
                seen[condition].add(data_idx)
                torch.cuda.empty_cache()

        counts = ", ".join(f"{name}={len(seen[name].intersection(qa_indices))}" for name in conditions)
        tqdm.write(f"{task_i:02d} {task_type}: selected={len(qa_indices)}; completed {counts}")


if __name__ == "__main__":
    main()
