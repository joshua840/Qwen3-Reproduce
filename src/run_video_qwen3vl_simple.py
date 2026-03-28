import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import argparse
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from vlmeval.smp import *
from vlmeval.dataset.videomme import VideoMMEDataset
from vlmeval.dataset.mvbench import MVBenchDataset
from vlmeval.dataset.mlvu import MLVUDataset

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_videomme_messages(data_root, line, total_pixels=56000, max_frames=512):
    video_path = osp.join(data_root, line['video_path'])
    question = line['question'] + '\n' + '\n'.join(eval(line['candidates']))
    return [{'role': 'user', 'content': [
        {'type': 'video', 'video': video_path,
         'min_pixels': 128*32*32, 'max_pixels': 768*32*32,
         'total_pixels': total_pixels*32*32, 'max_frames': max_frames, 'fps': 2},
        {'type': 'text', 'text': VideoMMEDataset.FRAMES_TMPL_NOSUB},
        {'type': 'text', 'text': f'Question: {question}\nAnswer: '},
    ]}]


def build_mvbench_messages(data_root, line, total_pixels=56000, max_frames=512):
    question = f"Question: {line['question']}\nOptions:\n"
    for idx, c in enumerate(eval(line['candidates'])):
        question += f"({chr(ord('A') + idx)}) {c}\n"
    question = question.rstrip()
    video_path = os.path.join(data_root, line['prefix'], line['video'])
    return [
        {'role': 'system', 'content': MVBenchDataset.SYS},
        {'role': 'user', 'content': [
            {'type': 'video', 'video': video_path,
             'min_pixels': 128*32*32, 'max_pixels': 768*32*32,
             'total_pixels': total_pixels*32*32, 'max_frames': max_frames, 'fps': 2},
            {'type': 'text', 'text': question},
            {'type': 'text', 'text': '\nOnly give the best option.'},
        ]},
    ]


def build_mlvu_messages(data_root, line, total_pixels=56000, max_frames=512):
    question = f"Question: {line['question']}\nOptions:\n"
    for idx, c in enumerate(eval(line['candidates'])):
        question += f"({chr(ord('A') + idx)}) {c}\n"
    question = question.rstrip()
    video_path = os.path.join(data_root, line['prefix'], line['video'])
    return [
        {'role': 'system', 'content': MLVUDataset.SYS},
        {'role': 'user', 'content': [
            {'type': 'video', 'video': video_path,
             'min_pixels': 128*32*32, 'max_pixels': 768*32*32,
             'total_pixels': total_pixels*32*32, 'max_frames': max_frames, 'fps': 2},
            {'type': 'text', 'text': question},
            {'type': 'text', 'text': '\nOnly give the best option.'},
        ]},
    ]


DATASETS = {
    'Video-MME': {
        'hf_repo': 'lmms-lab/Video-MME',
        'branch': None,
        'tsv': 'Video-MME.tsv',
        'build_messages': build_videomme_messages,
        'evaluate': VideoMMEDataset.evaluate,
    },
    'MVBench': {
        'hf_repo': 'OpenGVLab/MVBench',
        'branch': 'video',
        'tsv': 'MVBench_MP4.tsv',
        'build_messages': build_mvbench_messages,
        'evaluate': MVBenchDataset.evaluate,
    },
    'MLVU': {
        'data_root': '/data/MLVU',
        'tsv': 'MLVU_MCQ.tsv',
        'build_messages': build_mlvu_messages,
        'evaluate': MLVUDataset.evaluate,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--dataset', type=str, default='Video-MME')
    parser.add_argument('--output', type=str, default='./outputs/Qwen3-VL-simple')
    parser.add_argument('--total_pixels', type=int, default=56000)
    parser.add_argument('--max_frames', type=int, default=512)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    if 'data_root' in cfg:
        data_root = cfg['data_root']
        assert osp.exists(osp.join(data_root, cfg['tsv'])), f"Dataset not found at {data_root}"
    else:
        data_root = get_cache_path(cfg['hf_repo'], branch=cfg.get('branch')) if cfg.get('branch') else get_cache_path(cfg['hf_repo'])
        assert data_root, f"Dataset not found. Run: huggingface-cli download {cfg['hf_repo']} --repo-type dataset"
    data = pd.read_csv(osp.join(data_root, cfg['tsv']), sep='\t')

    model_name = osp.basename(args.model_path)
    pred_root = osp.join(args.output, model_name)
    os.makedirs(pred_root, exist_ok=True)
    result_path = osp.join(pred_root, f'{model_name}_{args.dataset}_tp{args.total_pixels}_mf{args.max_frames}.jsonl')

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        device_map='auto', attn_implementation='flash_attention_2',
    )

    with torch.inference_mode():
        for idx in tqdm(range(len(data)), desc=args.dataset):
            line = data.iloc[idx]
            messages = cfg['build_messages'](data_root, line, total_pixels=args.total_pixels, max_frames=args.max_frames)

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            inputs = processor(
                text=text, images=images, videos=videos,
                video_metadata=video_metadatas, return_tensors='pt',
                do_resize=False, **video_kwargs,
            ).to(model.device)

            generated_ids = model.generate(
                **inputs, do_sample=False, max_new_tokens=16,
                temperature=None, top_p=None, top_k=None,
            )
            output_text = processor.decode(
                generated_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            del inputs, generated_ids
            torch.cuda.empty_cache()

            line = line.to_dict()
            line['prediction'] = output_text
            with open(result_path, 'a', encoding='utf-8') as f:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

    logger.info(f'Inference done: {result_path}')
    rating = cfg['evaluate'](result_path)
    logger.info(json.dumps(rating, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
