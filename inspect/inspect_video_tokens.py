"""Inspect how video frames are reshaped and tokens are formed for different max_frames/total_pixels settings.

Usage:
    python inspect/inspect_video_tokens.py --video_path /mnt/ssd/data/huggingface/hub/datasets--longvideobench--LongVideoBench/snapshots/60d1c89c1919a198b73be39c2babb213b29d6a5c/videos/005BeD0c2PA.mp4
"""
import logging
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import fire
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import smart_resize, smart_nframes

# Suppress warnings/logs during processing
logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')


def inspect(
    video_path: str,
    model_path: str = 'Qwen/Qwen3-VL-2B-Instruct',
    max_frames_list: str = '16,32,64,128,256,512',
    total_pixels_list: str = '',
):
    """Analyze frame sampling, resize dimensions, and token counts for a video.

    If total_pixels_list is given, iterates over the cross product of max_frames_list x total_pixels_list.
    Otherwise, total_pixels is auto-computed from max_frames (max_frames * 109.375).
    """
    max_frames_list = [int(x) for x in max_frames_list.split(',')]
    total_pixels_list = [int(x) for x in total_pixels_list.split(',')] if total_pixels_list else []

    processor = AutoProcessor.from_pretrained(model_path)
    patch_size = processor.image_processor.patch_size  # 16
    merge_size = processor.video_processor.merge_size  # 2
    temporal_patch_size = processor.video_processor.temporal_patch_size  # 2
    factor = patch_size * merge_size  # 32

    # Get video info
    import decord
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration = total_frames / video_fps
    orig_h, orig_w = vr[0].shape[:2]

    print(f'Video: {video_path}')
    print(f'patch_size={patch_size}, merge_size={merge_size}, temporal_patch_size={temporal_patch_size}, factor={factor}')
    print(f'Original: {total_frames} frames, {video_fps:.1f} fps, {duration:.1f}s, {orig_w}x{orig_h}')
    print()

    header = f'{"max_frames":>10} {"total_pixels":>12} {"sampled_f":>9} {"resize_WxH":>12} {"grid_TxHxW":>12} {"patches":>8} {"tokens":>7} {"token_ratio":>11}'
    print(header)
    print('-' * len(header))

    # Build (max_frames, total_pixels) pairs
    if total_pixels_list:
        pairs = [(mf, tp) for mf in max_frames_list for tp in total_pixels_list]
    else:
        pairs = [(mf, int(mf * 109.375)) for mf in max_frames_list]

    for mf, tp in pairs:

        # Build message
        content = [
            {
                'type': 'video',
                'video': video_path,
                'min_pixels': 128 * factor * factor,
                'max_pixels': 768 * factor * factor,
                'total_pixels': tp * factor * factor,
                'max_frames': mf,
                'fps': 2,
            },
            {'type': 'text', 'text': 'Describe this video.'},
        ]
        messages = [{'role': 'user', 'content': content}]

        # Process vision info (frame sampling + resize)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            video_tensors, video_metadatas = zip(*videos)
            video_tensors = list(video_tensors)
            video_metadatas = list(video_metadatas)
        else:
            continue

        v = video_tensors[0]  # (T, C, H, W)
        sampled_frames = v.shape[0]
        resized_h, resized_w = v.shape[2], v.shape[3]

        # Process through video processor to get grid
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text, images=images, videos=video_tensors,
            video_metadata=video_metadatas, return_tensors="pt",
            do_resize=False, **video_kwargs,
        )

        grid_thw = inputs.get('video_grid_thw')
        if grid_thw is not None:
            grid_t, grid_h, grid_w = grid_thw[0].tolist()
        else:
            grid_t = sampled_frames // temporal_patch_size
            grid_h = resized_h // patch_size
            grid_w = resized_w // patch_size

        num_patches = int(grid_t * grid_h * grid_w)
        num_tokens = num_patches // (merge_size ** 2)

        # Token ratio relative to total input tokens
        total_input_tokens = inputs['input_ids'].shape[1]
        token_ratio = num_tokens / total_input_tokens

        resize_str = f'{resized_w}x{resized_h}'
        grid_str = f'{int(grid_t)}x{int(grid_h)}x{int(grid_w)}'
        print(
            f'{mf:>10} {tp:>12} {sampled_frames:>9} '
            f'{resize_str:>12} {grid_str:>12} '
            f'{num_patches:>8} {num_tokens:>7} {token_ratio:>10.1%}'
        )

        del inputs, video_tensors, video_metadatas

    print("""
Legend:
  sampled_f   = frames sampled from video (after fps=2 sampling, clamped by max_frames)
  resize_WxH  = frame dimensions after smart_resize
  grid_TxHxW  = patch grid (T/temporal_patch, H/patch, W/patch)
  patches     = grid_T * grid_H * grid_W (raw patch count)
  tokens      = patches / merge_size^2 (visual tokens fed to LLM)
  token_ratio = visual tokens / total input tokens""")


if __name__ == '__main__':
    fire.Fire(inspect)
