export HF_HOME=~/.cache/huggingface

# Qwen2.5-VL-7B-Instruct
# total_pixels = max_frames * 32  (paper: tp=24576, mf=768)
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 512 --max_frames 16
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 1024 --max_frames 32
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 2048 --max_frames 64
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 4096 --max_frames 128
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 8192 --max_frames 256
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 16384 --max_frames 512
CUDA_VISIBLE_DEVICES=0 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 24576 --max_frames 768

CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 512 --max_frames 16
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 1024 --max_frames 32
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 2048 --max_frames 64
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 4096 --max_frames 128
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 8192 --max_frames 256
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 16384 --max_frames 512
CUDA_VISIBLE_DEVICES=1 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 24576 --max_frames 768

CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 512 --max_frames 16
CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 1024 --max_frames 32
CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 2048 --max_frames 64
CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 4096 --max_frames 128
CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 8192 --max_frames 256
CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 16384 --max_frames 512
CUDA_VISIBLE_DEVICES=2 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-7B-Instruct  --total_pixels 24576 --max_frames 768

# Qwen2.5-VL-3B-Instruct
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 512 --max_frames 16
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 1024 --max_frames 32
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 2048 --max_frames 64
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 4096 --max_frames 128
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 8192 --max_frames 256
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 16384 --max_frames 512
CUDA_VISIBLE_DEVICES=3 python src/run_video_qwen25vl.py --dataset Video-MME --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 24576 --max_frames 768

CUDA_VISIBLE_DEVICES=4 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 512 --max_frames 16
CUDA_VISIBLE_DEVICES=4 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 1024 --max_frames 32
CUDA_VISIBLE_DEVICES=4 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 2048 --max_frames 64
CUDA_VISIBLE_DEVICES=4 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 4096 --max_frames 128
CUDA_VISIBLE_DEVICES=4 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 8192 --max_frames 256
CUDA_VISIBLE_DEVICES=4 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 16384 --max_frames 512
CUDA_VISIBLE_DEVICES=4 python src/run_video_qwen25vl.py --dataset MLVU --data_root /data/MLVU --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 24576 --max_frames 768

CUDA_VISIBLE_DEVICES=5 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 512 --max_frames 16
CUDA_VISIBLE_DEVICES=5 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 1024 --max_frames 32
CUDA_VISIBLE_DEVICES=5 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 2048 --max_frames 64
CUDA_VISIBLE_DEVICES=5 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 4096 --max_frames 128
CUDA_VISIBLE_DEVICES=5 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 8192 --max_frames 256
CUDA_VISIBLE_DEVICES=5 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 16384 --max_frames 512
CUDA_VISIBLE_DEVICES=5 python src/run_video_qwen25vl.py --dataset LongVideoBench --data_root /mnt/ssd/data/LongVideoBench --model_path Qwen/Qwen2.5-VL-3B-Instruct  --total_pixels 24576 --max_frames 768
