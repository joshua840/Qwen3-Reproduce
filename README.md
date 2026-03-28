# Qwen3-VL Video Benchmark Reproduction

Minimal codebase for reproducing Qwen3-VL results on video understanding benchmarks (Video-MME, MLVU, MVBench).

Evaluation utilities are extracted from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Structure

```
src/
  run_video_qwen3vl.py          # KV cache prefix sharing (fast)
  run_video_qwen3vl_simple.py   # Independent inference per question (simple)
  vlmeval/                      # Minimal VLMEvalKit dependencies
scripts/
  run.sh                        # Example launch commands
check_outputs.ipynb             # Results analysis notebook
outputs/                        # Experiment results
```

## `run_video_qwen3vl.py` vs `run_video_qwen3vl_simple.py`

| | `run_video_qwen3vl.py` | `run_video_qwen3vl_simple.py` |
|---|---|---|
| Video processing | Shares KV cache prefix across questions of the same video | Re-encodes video for each question |
| Speed | Fast (1 video encoding per video) | Slow (1 video encoding per question) |
| Complexity | Higher (DynamicCache management) | Lower (straightforward generate) |

## Setup

```bash
# Install project dependencies
uv sync

# Install flash-attn separately (requires CUDA toolkit)
uv pip install flash-attn --no-build-isolation
```

## Dataset Preparation

- **Video-MME**: `huggingface-cli download lmms-lab/Video-MME --repo-type dataset`
- **MVBench**: `huggingface-cli download OpenGVLab/MVBench --repo-type dataset --revision video`
- **MLVU**: Place manually at `/data/MLVU` (or modify the path in code)

## Usage

```bash
# Single run
CUDA_VISIBLE_DEVICES=0 uv run src/run_video_qwen3vl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset Video-MME \
    --total_pixels 56000 \
    --max_frames 512

# Parallel runs
bash scripts/run.sh
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-VL-2B-Instruct` | HuggingFace model path |
| `--dataset` | `Video-MME` | `Video-MME` / `MLVU` / `MVBench` |
| `--output` | `./outputs/Qwen3-VL` | Output directory |
| `--total_pixels` | `224000` | Total pixels for video frames |
| `--max_frames` | `2048` | Maximum number of frames |

`total_pixels` and `max_frames` scale proportionally:

| total_pixels | max_frames |
|---|---|
| 28000 | 256 |
| 56000 | 512 |
| 112000 | 1024 |
| 224000 | 2048 |

## Results

Results are saved to `outputs/`. Use `check_outputs.ipynb` to analyze.

## Acknowledgments

Evaluation code is from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
