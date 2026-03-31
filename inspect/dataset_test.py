"""
데이터셋 흐름 파악용 테스트 코드
- VideoMME, MLVU 데이터셋 객체 생성
- processor 생성
- inference 파이프라인의 각 함수 input/output 확인 (model 제외)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from dataset.videomme import VideoMMEDataset
from dataset.mlvu import MLVUDataset


MODEL_PATH = 'Qwen/Qwen3-VL-2B-Instruct'
TOTAL_PIXELS = 3500
MAX_FRAMES = 32


def inspect_tensor(name, t):
    if isinstance(t, torch.Tensor):
        print(f"  {name}: shape={t.shape}, dtype={t.dtype}")
    elif isinstance(t, list):
        print(f"  {name}: list, len={len(t)}")
        for i, item in enumerate(t[:2]):
            if isinstance(item, torch.Tensor):
                print(f"    [{i}]: shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, dict):
                print(f"    [{i}]: dict, keys={list(item.keys())}")
            else:
                print(f"    [{i}]: {type(item).__name__}")
    elif t is None:
        print(f"  {name}: None")
    else:
        print(f"  {name}: {type(t).__name__} = {repr(t)[:500]}")


def test_dataset(dataset_cls, dataset_name):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    # 1. Dataset 생성
    kwargs = dict(total_pixels=TOTAL_PIXELS, max_frames=MAX_FRAMES)
    if dataset_cls == MLVUDataset:
        kwargs['data_root'] = '/data/MLVU'
    dataset = dataset_cls(**kwargs)
    print(f"\ndata shape: {dataset.data.shape}")
    print(f"data columns: {list(dataset.data.columns)}")
    print(f"num videos (groups): {len(dataset.groups)}")
    print(f"group[0] shape: {dataset.groups[0].shape}")

    # 2. __getitem__ → lines, messages_list
    lines, messages_list = dataset[0]
    print(f"\n--- dataset[0] ---")
    print(f"lines: {len(lines)} questions")
    print(f"messages_list: {len(messages_list)} messages")

    print(f"\nlines[0] keys: {list(lines[0].index)}")
    print(f"lines[0]:")
    for k, v in lines[0].items():
        print(f"  {k}: {repr(v)[:200]}")

    print(f"\nmessages_list[0] (chat format):")
    for msg in messages_list[0]:
        role = msg['role']
        content = msg['content']
        if isinstance(content, str):
            print(f"  [{role}]: {content[:500]}")
        elif isinstance(content, list):
            print(f"  [{role}]: list of {len(content)} items")
            for item in content:
                t = item.get('type', '?')
                if t == 'text':
                    print(f"    - text: {item['text'][:500]}")
                elif t == 'video':
                    print(f"    - video: {item.get('video', '?')}")
                    for k, v in item.items():
                        if k not in ('type', 'video'):
                            print(f"      {k}: {v}")

    return dataset, lines, messages_list


def test_processor(processor, messages_list):
    print(f"\n{'='*60}")
    print(f"Processor pipeline")
    print(f"{'='*60}")

    # 3. process_vision_info
    print(f"\n--- process_vision_info ---")
    print(f"input: messages_list[0] (chat messages)")
    images, videos, video_kwargs = process_vision_info(
        messages_list[0],
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    print(f"output:")
    inspect_tensor("images", images)
    inspect_tensor("videos (raw)", videos)
    inspect_tensor("video_kwargs", video_kwargs)

    # Unpack video metadata
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    print(f"\nAfter unpack:")
    inspect_tensor("videos", videos)
    inspect_tensor("video_metadatas", video_metadatas)
    if videos:
        print(f"  videos[0]: shape={videos[0].shape}, dtype={videos[0].dtype}")
    if video_metadatas:
        print(f"  video_metadatas[0]: {video_metadatas[0]}")

    # 4. apply_chat_template
    print(f"\n--- apply_chat_template ---")
    first_text = processor.apply_chat_template(
        messages_list[0], tokenize=False, add_generation_prompt=True
    )
    print(f"output type: str, len={len(first_text)}")
    print(f"first chars:\n{first_text}")

    # 5. processor() — full processing
    print(f"\n--- processor() (full) ---")
    base_inputs = processor(
        text=first_text, images=images, videos=videos,
        video_metadata=video_metadatas, return_tensors="pt",
        do_resize=False, **video_kwargs
    )
    print(f"output keys: {list(base_inputs.keys())}")
    for k, v in base_inputs.items():
        inspect_tensor(k, v)

    # 6. Prefix split (vision_end)
    print(f"\n--- Prefix split ---")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    print(f"vision_end_id: {vision_end_id}")
    prefix_len = (base_inputs.input_ids[0] == vision_end_id).nonzero()[-1].item() + 1
    seq_len = base_inputs.input_ids.shape[1]
    print(f"prefix_len: {prefix_len}")
    print(f"seq_len: {seq_len}")
    print(f"suffix_len (first question): {seq_len - prefix_len}")

    # 7. Second question suffix (if multiple questions)
    if len(messages_list) > 1:
        print(f"\n--- Second question suffix ---")
        text2 = processor.apply_chat_template(
            messages_list[1], tokenize=False, add_generation_prompt=True
        )
        vision_end_marker = '<|vision_end|>'
        suffix_text = text2[text2.rfind(vision_end_marker) + len(vision_end_marker):]
        suffix_ids = processor.tokenizer.encode(
            suffix_text, return_tensors='pt', add_special_tokens=False
        )
        print(f"suffix_text (first 200): {suffix_text[:500]}")
        print(f"suffix_ids shape: {suffix_ids.shape}")


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"Processor loaded: {MODEL_PATH}")
    print(f"patch_size: {processor.image_processor.patch_size}")

    for dataset_cls, name in [(VideoMMEDataset, 'Video-MME'), (MLVUDataset, 'MLVU')]:
        dataset, lines, messages_list = test_dataset(dataset_cls, name)
        test_processor(processor, messages_list)
