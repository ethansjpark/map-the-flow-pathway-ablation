# Vast.ai RTX 4090 run sequence

The original run used a Vast.ai RTX 4090, the `pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel` image, and an 80 GB disk. Commands below assume the package is available on the instance at `/workspace/map-the-flow-pathway-ablation`. Run steps 1–6 from its `scripts/` directory; return to the package root for step 7. The scripts use the upstream checkout at `/workspace/map-the-flow`.

1. Install the Hub client:

   ```bash
   pip install huggingface_hub==0.21.4
   ```

2. Prepare FlashAttention:

   ```bash
   pip install ninja && pip install flash-attn==2.5.6 --no-build-isolation
   ```

3. Clone the upstream repository at commit `566f3ed064d324408fdf96f7a921ebf1e65f0908`, install pinned requirements, then install `mmcv-full`:

   ```bash
   bash setup_env.sh
   ```

4. Download `byminji/LLaVA-NeXT-7B-Video-FT`, TVBench JSON, and four task archives:

   ```bash
   bash download_assets.sh
   ```

5. Check the instance and run one Object Count smoke sample:

   ```bash
   bash check_env.sh ; bash smoke_test.sh
   ```

6. Run all questions in four TVBench tasks with `effective-pathway-7b` and random blocking at ratio 0.58:

   ```bash
   bash run_reproduction.sh
   ```

7. From the package root, run the three-task ablation and matched 33% random control, then summarize each result directory:

   ```bash
   python3 pathway_ablation/pathway_ablation.py --output_dir /workspace/outputs/ablation --model_path /workspace/models/LLaVA-NeXT-7B-Video-FT --task_ids 3,6,8 --n_per_task 150
   python3 pathway_ablation/pathway_ablation.py --output_dir /workspace/outputs/ablation_random33 --model_path /workspace/models/LLaVA-NeXT-7B-Video-FT --task_ids 3,6,8 --n_per_task 150 --conditions random --random_block_ratio 0.667
   python3 pathway_ablation/summarize_ablation.py /workspace/outputs/ablation
   python3 pathway_ablation/summarize_ablation.py /workspace/outputs/ablation_random33
   ```

To export the public reproduction results from the unstripped upstream JSON, run from the package root:

```bash
python3 scripts/export_reproduction_results.py /workspace/outputs/reproduction/tvbench results/reproduction
python3 pathway_ablation/summarize_ablation.py results/reproduction > results/reproduction/table.txt
```

The first server run installed FlashAttention in step 2 and removed the `apex==` requirement by hand. `setup_env.sh` now removes that line from the instance checkout before installing requirements; this scripted version was not re-run end to end **[UNTESTED]**. PyPI `apex` is unrelated to optional NVIDIA apex in the upstream code. Approximate RTX 4090 GPU time was 30 minutes for the reproduction and 45 minutes for the ablation.
