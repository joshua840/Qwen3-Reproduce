export HF_HOME=~/.cache/huggingface
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen3vl_simple.py --dataset Video-MME --model_path Qwen/Qwen3-VL-2B-Instruct  --total_pixels 1750 --max_frames 16 
