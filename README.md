# Qwen3-VL Video Benchmark Reproduction

Minimal codebase for reproducing Qwen3-VL results on video understanding benchmarks (Video-MME, MLVU, MVBench).

Evaluation utilities are adapted from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Structure

```
src/
  run_video_qwen3vl.py              # KV cache prefix sharing (fast)
  vlmeval/dataset/
    video_base.py                   # VideoDataset base class (prompt, scoring, evaluation)
    videomme.py                     # VideoMMEDataset
    mlvu.py                         # MLVUDataset
    mvbench.py                      # MVBenchDataset
scripts/
  run.sh                            # Example launch commands
outputs/
  check_outputs.ipynb               # Results analysis notebook
```

## Prompt Template

All datasets use the same MCQ prompt (Qwen3-VL official):

```
<video>
Select the best answer to the following multiple-choice question based on the video.
Respond with only the letter (A, B, C, or D) of the correct option.
Question: {question} Possible answer choices:
(A) ...
(B) ...
(C) ...
(D) ...
The best answer is:
```

## Setup

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

## Dataset Preparation

**Video-MME** (auto-detected from HF cache):
```bash
huggingface-cli download lmms-lab/Video-MME --repo-type dataset

# Unzip video files in the snapshot directory
cd ~/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/<hash>/
mkdir -p video && for f in videos_chunked_*.zip; do unzip -jo "$f" -d video/; done
```

**MLVU** (auto-detected from HF cache):
```bash
huggingface-cli download MLVU/MVLU --repo-type dataset
```

**MVBench** (auto-detected from HF cache):
```bash
huggingface-cli download OpenGVLab/MVBench --repo-type dataset --revision video
```

All datasets can alternatively use `--data_root` to point to a local directory.

## Usage

```bash
# Single run
CUDA_VISIBLE_DEVICES=0 uv run src/run_video_qwen3vl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset Video-MME \
    --total_pixels 56000 \
    --max_frames 512

# MLVU with local data
CUDA_VISIBLE_DEVICES=0 uv run src/run_video_qwen3vl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset MLVU \
    --data_root /data/MLVU \
    --total_pixels 28000 \
    --max_frames 256

# Parallel runs
bash scripts/run.sh
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-VL-2B-Instruct` | HuggingFace model path |
| `--dataset` | `Video-MME` | `Video-MME` / `MLVU` / `MVBench` |
| `--data_root` | `None` (auto-detect HF cache) | Path to dataset root directory |
| `--output` | `./outputs` | Output directory |
| `--total_pixels` | `224000` | Total pixels for video frames |
| `--max_frames` | `2048` | Maximum number of frames |

`total_pixels` and `max_frames` scale proportionally:

| total_pixels | max_frames |
|---|---|
| 28000 | 256 |
| 56000 | 512 |
| 112000 | 1024 |
| 224000 | 2048 |

## Output Structure

Results are organized per experiment:

```
outputs/
  {model}/
    {dataset}/
      tp{total_pixels}_mf{max_frames}/
        predictions.jsonl       # Raw model predictions
        score.json              # Scored predictions
        rating.json             # Per-dimension accuracy breakdown
```

Use `outputs/check_outputs.ipynb` to analyze results.

## Adding a New Dataset

Subclass `VideoDataset` and implement 4 methods:

```python
from .video_base import VideoDataset, get_cache_path

class NewDataset(VideoDataset):

    def _default_data_root(self):
        root = get_cache_path('org/repo')
        if root is not None:
            return root
        raise FileNotFoundError('...')

    def _load_data(self):
        # Return a DataFrame with columns: video, question, candidates, answer, task_type, index
        ...

    def _get_video_path(self, line):
        # Return absolute path to video file
        ...

    @classmethod
    def _get_dimension_rating(cls, data):
        # Return rating dict from scored DataFrame
        ...
```

The base class handles: prompt building, message formatting, scoring (`extract_letter`), evaluation, and file I/O.

Then register in `run_video_qwen3vl.py`:
```python
from vlmeval.dataset.new_dataset import NewDataset
DATASET_MAP['NewDataset'] = NewDataset
```

## Acknowledgments

Evaluation code is adapted from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
