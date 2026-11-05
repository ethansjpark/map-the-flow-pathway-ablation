# Credits and provenance

This package reproduces and extends **Map the Flow** by Kim, Kim, and Han (ICLR 2026; [arXiv:2510.13251](https://arxiv.org/abs/2510.13251)). The upstream repository is [byminji/map-the-flow](https://github.com/byminji/map-the-flow), pinned at commit `566f3ed064d324408fdf96f7a921ebf1e65f0908` for the reproduction.

The upstream repository had no license file as of this packaging. No upstream source files are redistributed here. `pathway_ablation/pathway_ablation.py` adapts logic from its `analysis/effective_pathway_analysis.py`; the other packaged scripts and results are from this independent reproduction effort.

The model is [byminji/LLaVA-NeXT-7B-Video-FT](https://huggingface.co/byminji/LLaVA-NeXT-7B-Video-FT). The dataset is [FunAILab/TVBench](https://huggingface.co/datasets/FunAILab/TVBench). Results in this repository contain no question text or videos. Follow the dataset's own license notice when obtaining or using it.

Not affiliated with the original authors. Happy to remove or change anything on request. Contact: GitHub [@ethansjpark](https://github.com/ethansjpark) only.
