# transformers backend, on 8 gpus, get acc round 71.3
export HF_HOME=~/.cache/huggingface
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen3vl.py --dataset Video-MME --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 28000 --max_frames 256 &
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl.py --dataset Video-MME --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 56000 --max_frames 512 &

CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen3vl.py --dataset MLVU --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 28000 --max_frames 256 &
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen3vl.py --dataset MLVU --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 56000 --max_frames 512 &

CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen3vl.py --dataset Video-MME --model_path Qwen/Qwen3-VL-4B-Instruct  --total_pixels 28000 --max_frames 256 &
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl.py --dataset Video-MME --model_path Qwen/Qwen3-VL-4B-Instruct  --total_pixels 56000 --max_frames 512 &

CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen3vl.py --dataset MLVU --model_path Qwen/Qwen3-VL-4B-Instruct  --total_pixels 28000 --max_frames 256 &
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen3vl.py --dataset MLVU --model_path Qwen/Qwen3-VL-4B-Instruct  --total_pixels 56000 --max_frames 512 &

# Simple code
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen3vl_simple.py --dataset Video-MME --model_path Qwen/Qwen3-VL-2B-Instruct --total_pixels 56000 --max_frames 512 &
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen3vl_simple.py --dataset MLVU --model_path Qwen/Qwen3-VL-2B-Instruct --total_pixels 56000 --max_frames 512 &
