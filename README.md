<div align="center">

# GRASP: Learning to Ground Social Reasoning in Multi-Person Non-Verbal Interactions

**Advances in Neural Information Processing Systems (NeurIPS), 2026**

[\[📜 Paper\]](https://arxiv.org/abs/2605.15764)
[\[🌐 Project Page\]](https://social-reaoning.github.io/grasp/)
[\[🤗 Dataset\]](https://huggingface.co/datasets/interlive/GRASP)
[\[🤗 Model\]](https://huggingface.co/interlive/GRASP-Qwen3-VL-8B)

Junho Kim<sup>1</sup>, Xu Cao<sup>1</sup>, Houze Yang<sup>1</sup>, Bikram Boote<sup>1</sup>, Ana Jojic<sup>1</sup>,
Fiona Ryan<sup>2</sup>, Bolin Lai<sup>3</sup>, Sangmin Lee<sup>4</sup>, James M. Rehg<sup>1</sup>

<sup>1</sup>UIUC · <sup>2</sup>Georgia Tech · <sup>3</sup>Amazon AGI · <sup>4</sup>Korea University

</div>

## Introduction

**GRASP** is a large-scale social reasoning dataset that connects high-level social QA with fine-grained gaze and deictic gesture events — 290K question–answer pairs over 46K videos totaling 749 hours, organized by a 16-category taxonomy spanning gaze, gesture, and joint gaze–gesture reasoning, together with **GRASP-Bench** for evaluation. We also propose **Social Grounding Reward (SGR)**, a learning signal that uses these social events to encourage models to reason about the participants involved in each interaction.

## TODO

- [x] Paper release
- [x] Project page
- [x] Training code (SFT + SGR)
- [x] GRASP dataset and GRASP-Bench
- [x] Model weights (GRASP-Qwen3-VL-8B)

## Setup

Requires Python 3.12 and CUDA 12.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Supported models: `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.5-9B`.

## Data

Download [interlive/GRASP](https://huggingface.co/datasets/interlive/GRASP) and extract the video archives:

```bash
hf download interlive/GRASP --repo-type dataset --local-dir data/GRASP
mkdir -p data/videos data/grasp_bench
for t in data/GRASP/train/videos/*.tar; do tar -xf "$t" -C data/videos; done
tar -xf data/GRASP/test/grasp_bench.tar -C data/grasp_bench
```

## Training

Default hyperparameters follow the paper (SFT) and the released checkpoint (SGR). Override any of the variables at the top of each script via environment variables.

### 1. SFT

```bash
DATA_PATH=data/GRASP/train/data_sft.jsonl VIDEO_ROOT=data/videos \
bash scripts/train_sft.sh
```

### 2. SGR (GRPO with Social Grounding Reward)

```bash
DATA_PATH=data/GRASP/train/data_rl.jsonl VIDEO_ROOT=data/videos \
MODEL_PATH=checkpoints/grasp_sft \
bash scripts/train_sgr.sh
```

The reward combines answer correctness, response format, and participant-level grounding (`correct` / `has_format` / `has_grounding` / `grounding_quality` = 1.0 / 0.1 / 0.05 / 0.2), computed against the ground-truth social events (`gt_events`) attached to each MCQ sample.

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | HuggingFace model ID or local checkpoint |
| `MODEL_TYPE` | `qwen3_vl`, `qwen3_5` |
| `DATA_PATH` | Training JSONL |
| `VIDEO_ROOT` | Directory the video tars were extracted into |
| `NUM_GPUS` | GPUs for DeepSpeed ZeRO-2 (default: 8) |
| `OUTPUT_DIR` | Checkpoint output directory |

## Evaluation

```bash
BENCH_DIR=data/grasp_bench MODEL_PATH=interlive/GRASP-Qwen3-VL-8B \
bash scripts/eval_grasp_bench.sh
```

Per-category accuracy (Gaze / Gesture / Joint) is printed at the end and predictions are saved under `eval/results/`.

## License

Code, annotations, and model weights are released under [CC BY-NC 4.0](LICENSE) for non-commercial research use. Video clips remain subject to the licenses of their original source datasets.

## Citation

```bibtex
@article{kim2026grasp,
  title={GRASP: Learning to Ground Social Reasoning in Multi-Person Non-Verbal Interactions},
  author={Kim, Junho and Cao, Xu and Yang, Houze and Boote, Bikram and Jojic, Ana and Ryan, Fiona and Lai, Bolin and Lee, Sangmin and Rehg, James M},
  journal={arXiv preprint arXiv:2605.15764},
  year={2026}
}
```
