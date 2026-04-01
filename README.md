# Qwen3-VL Video Benchmark Reproduction

Minimal codebase for reproducing Qwen3-VL results on video understanding benchmarks (Video-MME, MLVU, MVBench).

Evaluation utilities are adapted from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Structure

```
src/
  run_video_qwen3vl.py              # KV cache prefix sharing (fast)
  dataset/
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

## Performance

Accuracy (%) by number of input frames (`max_frames`):

| Model | Dataset | mf=16 | mf=32 | mf=64 | mf=128 | mf=256 | mf=512 | Report (mf=2048) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | MLVU | 55.70 | 61.04 | 63.80 | 67.76 | 69.50 | 70.29 | 68.3 |
| Qwen3-VL-2B-Instruct | Video-MME | 53.81 | 57.00 | 59.19 | 61.70 | 61.96 | 63.19 | 61.9 |
| Qwen3-VL-4B-Instruct | MLVU | 58.05 | 62.33 | 67.07 | 70.10 | 74.89 | - | 75.3 |
| Qwen3-VL-4B-Instruct | Video-MME | 56.33 | 60.59 | 63.07 | 66.11 | 67.15 | - | 69.3 |
| Qwen3-VL-2B-Instruct | LongVideoBench | 50.93 | 52.36 | 54.97 | 57.37 | 58.56 | - | - |
| Qwen3-VL-4B-Instruct | LongVideoBench | 55.87 | 59.16 | 60.88 | 63.65 | - | - | - |

mf=16 ~ mf=512: reproduced in this repo. Report: from [Qwen3-VL technical report](https://qwenlm.github.io/blog/qwen3-vl/) (mf=2048).

## Setup

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

## Dataset Preparation

If you download datasets via `huggingface-cli download`, they are **automatically detected** from the HF cache — no `--data_root` needed. If you already have the data stored locally, use `--data_root` to point to it (see [Usage](#usage)).

### Video-MME (~95GB)

```bash
huggingface-cli download lmms-lab/Video-MME --repo-type dataset

# Unzip video files in the snapshot directory
cd ~/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/<hash>/
mkdir -p video && for f in videos_chunked_*.zip; do unzip -jo "$f" -d video/; done
```

Expected structure (HF cache or `--data_root`):
```
Video-MME/              # data_root
  video/                # 900 .mp4 files
  videomme/             # metadata (parquet)
```

### MLVU (~400GB)

```bash
huggingface-cli download MLVU/MVLU --repo-type dataset
```

Expected structure (HF cache `MLVU/` subdir or `--data_root`):
```
MLVU/                   # data_root
  json/                 # metadata (1_plotQA.json, ..., 7_topic_reasoning.json)
  video/                # video files organized by task
    1_plotQA/           # 349 videos
    2_needle/           # 150 videos
    3_ego/              # 84 videos
    4_count/            # 239 videos
    5_order/            # 330 videos
    6_anomaly_reco/     # 200 videos
    7_topic_reasoning/  # 204 videos
```

### LongVideoBench (~130GB)

```bash
huggingface-cli download longvideobench/LongVideoBench --repo-type dataset

# Merge and extract video tar parts in the snapshot directory
cd ~/.cache/huggingface/hub/datasets--longvideobench--LongVideoBench/snapshots/<hash>/
cat videos.tar.part.* | tar xf -

# (Optional) To save disk space, convert symlinks to real files and delete blobs:
for f in lvb_val.json lvb_test_wo_gt.json subtitles.tar *.parquet README.md; do
  [ -L "$f" ] && cp --remove-destination "$(readlink -f "$f")" "$f"
done
# rm -rf ../../blobs
```

Expected structure (HF cache or `--data_root`):
```
LongVideoBench/         # data_root
  lvb_val.json          # validation metadata (1337 questions)
  videos/               # video files (.mp4)
```

### MVBench (currently not available — HF dataset is broken)

```bash
# huggingface-cli download OpenGVLab/MVBench --repo-type dataset --revision video
```

## Usage

```bash
# If downloaded via huggingface-cli, just run (auto-detected):
CUDA_VISIBLE_DEVICES=0 uv run src/run_video_qwen3vl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset Video-MME \
    --total_pixels 56000 \
    --max_frames 512

# If you have data stored locally, use --data_root:
CUDA_VISIBLE_DEVICES=0 uv run src/run_video_qwen3vl.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset MLVU \
    --data_root /path/to/MLVU \
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

`total_pixels` and `max_frames` are always set proportionally with a fixed ratio: `total_pixels = max_frames * 109.375` (i.e. `1750 * n` total pixels corresponds to `16 * n` max frames).

| max_frames | total_pixels |
|---|---|
| 16 | 1,750 |
| 32 | 3,500 |
| 64 | 7,000 |
| 128 | 14,000 |
| 256 | 28,000 |
| 512 | 56,000 |
| 1024 | 112,000 |
| 2048 | 224,000 |

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
