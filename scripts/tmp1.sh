# transformers backend, on 8 gpus, get acc round 71.3
export HF_HOME=~/.cache/huggingface

CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 1750 --max_frames 16 
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 3500 --max_frames 32
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 7000 --max_frames 64
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 14000 --max_frames 128 
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 28000 --max_frames 256
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 56000 --max_frames 512 
