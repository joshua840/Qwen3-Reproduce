# Qwen3-VL Video Benchmark Reproduction

Minimal codebase for reproducing Qwen3-VL results on video understanding benchmarks (Video-MME, MLVU, MVBench).

Evaluation utilities are extracted from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Structure

```
src/
  run_video_qwen3vl.py              # KV cache prefix sharing (fast)
  run_video_qwen3vl_simple.py       # Independent inference per question (simple)
  vlmeval/
    smp/                            # Minimal utilities (load/dump, get_cache_path)
    dataset/
      video_base.py                 # VideoDataset base class
      videomme.py                   # VideoMMEDataset (data + prompt + evaluate)
      mvbench.py                    # MVBenchDataset
      mlvu.py                       # MLVUDataset
scripts/
  run.sh                            # Example launch commands
check_outputs.ipynb                 # Results analysis notebook
outputs/                            # Experiment results
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

## Adding a New Dataset from VLMEvalKit

This codebase was heavily simplified from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The following was removed:

- **API / Judge model layer** (`vlmeval/api/`, `judge_util.py`, `multiple_choice.py`) — all evaluation is exact matching only (MCQ)
- **Dataset preparation** (`prepare_dataset`, `prepare_tsv`, `save_video_frames`) — datasets are assumed to be pre-downloaded via `huggingface-cli`
- **Image processing utilities** (`vlm.py`, `tools.py`, frame extraction) — only video inputs are used
- **Unused dataset classes** (`MVBench`, `MLVU`, `MLVU_OpenEnded`, `ConcatVideoDataset`)

When porting a new VLMEvalKit dataset, you only need to implement these 3 things in a single file:

### 1. Dataset class (extends `VideoDataset`)

```python
class NewDataset(VideoDataset):
    SYS = "..."  # system prompt (if any)

    def __init__(self, **kwargs):
        data_root = get_cache_path('org/repo')  # or hardcode path
        data_path = osp.join(data_root, 'dataset.tsv')
        super().__init__(dataset_name='Name', data_root=data_root, data_path=data_path, **kwargs)

    def _build_struct(self, line):
        # Return list of dicts: [{'type': 'video', 'value': path}, {'type': 'text', 'value': prompt}]
        ...
```

`VideoDataset` (in `video_base.py`) handles: TSV loading, video grouping, pixel/frame config, message formatting.

### 2. Evaluate method (exact matching)

```python
    @classmethod
    def evaluate(cls, eval_file, **kwargs):
        data = load(eval_file)
        # Compare prediction vs answer, compute score
        # Return rating dict
```

From VLMEvalKit's evaluate methods, **strip out**:
- `build_judge()` / `model.working()` / `DEBUG_MESSAGE` — no judge LLM needed
- `extract_answer_from_item()` — replace with simple regex or string matching
- `check_ans_with_model()` — replace with the exact-match part only (the first few lines before `elif extract_answer_from_item(...)`)

### 3. Register in main script

In `run_video_qwen3vl.py`:
```python
from vlmeval.dataset.new_dataset import NewDataset
DATASET_MAP['NewDataset'] = NewDataset
```

### What to copy from VLMEvalKit

| Need | Where in VLMEvalKit | What to keep |
|---|---|---|
| Prompt constants | Class attributes (`SYS`, `FRAMES_TMPL_*`) | Copy as-is |
| Question formatting | `build_prompt()` or `qa_template()` | Adapt into `_build_struct()` |
| Score computation | `evaluate()` classmethod | Keep the scoring logic, remove judge fallback |
| Dimension ratings | `get_dimension_rating()` helper | Copy into the same file |
| TSV column names | `prepare_dataset()` → `generate_tsv()` | Note which columns exist (`video`, `question`, `candidates`, `answer`, `task_type`, etc.) |

### What NOT to copy

- `__init__` / `prepare_dataset` — replace with `get_cache_path()` + TSV path
- `save_video_frames` / `frame_paths` — not needed (videos are handled by `qwen_vl_utils`)
- `build_judge` / API wrappers — not needed for MCQ
- `supported_datasets` / `MODALITY` — not needed

## Acknowledgments

Evaluation code is adapted from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
