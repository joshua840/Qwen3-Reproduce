import os
import os.path as osp
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import argparse
import pandas as pd
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, DynamicCache
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from vlmeval.dataset.videomme import VideoMMEDataset
from vlmeval.dataset.mvbench import MVBenchDataset
from vlmeval.dataset.mlvu import MLVUDataset

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATASET_MAP = {
    'Video-MME': VideoMMEDataset,
    'MVBench': MVBenchDataset,
    'MLVU': MLVUDataset,
}


class Evaluator:
    def __init__(
            self,
            model_path='Qwen/Qwen3-VL-2B-Instruct',
            dataset_name='Video-MME',
            output_dir='outputs',
            total_pixels=224000,
            max_frames=2048,
        ):
        self.model_name = osp.basename(model_path)
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.total_pixels = total_pixels
        self.max_frames = max_frames

        logger.info(f'Model name: {self.model_name}\nModel path: {self.model_path}\nDataset name: {self.dataset_name}')

        self.pred_root = osp.join(output_dir, self.model_name)
        self.result_file_path = osp.join(
            self.pred_root,
            f'{self.model_name}_{self.dataset_name}_tp{self.total_pixels}_mf{self.max_frames}.jsonl'
        )
        os.makedirs(self.pred_root, exist_ok=True)

        if dataset_name not in DATASET_MAP:
            raise ValueError(f'Unsupported dataset: {dataset_name}. Choose from {list(DATASET_MAP.keys())}')
        self.VIDEO_DATASET_CLS = DATASET_MAP[dataset_name]

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.dataset = self.VIDEO_DATASET_CLS(total_pixels=total_pixels, max_frames=max_frames)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation="flash_attention_2",
        )

    @torch.inference_mode()
    def inference(self):
        logger.info(f'Start running {self.model_name} x {self.dataset_name}')

        # Resume: collect already-done indices from result file
        done_indices = set()
        if osp.exists(self.result_file_path):
            with open(self.result_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    done_indices.add(json.loads(line)['index'])
            if done_indices:
                logger.info(f'Resuming: {len(done_indices)} questions already done')

        device = self.model.device

        for video_idx in tqdm(range(len(self.dataset)), desc=f'Processing {self.dataset_name}'):
            # Skip entire video group if all its questions are done
            group = self.dataset.groups[video_idx]
            if done_indices and set(group['index'].values).issubset(done_indices):
                continue

            lines, messages_list = self.dataset[video_idx]

            # Decode video once from first question's messages
            images, videos, video_kwargs = process_vision_info(
                messages_list[0],
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            # First message: full processor (includes video pixel values)
            first_text = self.processor.apply_chat_template(messages_list[0], tokenize=False, add_generation_prompt=True)
            base_inputs = self.processor(
                text=first_text, images=images, videos=videos,
                video_metadata=video_metadatas, return_tensors="pt",
                do_resize=False, **video_kwargs
            )

            # Prefix = everything up to and including <|vision_end|>
            vision_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            prefix_len = (base_inputs.input_ids[0] == vision_end_id).nonzero()[-1].item() + 1

            # Prefill prefix → KV cache
            logger.info(f'Video {video_idx}: prefix_len={prefix_len}, seq_len={base_inputs.input_ids.shape[1]}')
            cache = DynamicCache()
            base_inputs = base_inputs.to(device)
            seq_len = base_inputs.input_ids.shape[1]
            prefix_fwd = {
                k: v[:, :prefix_len] if v.dim() >= 2 and v.shape[1] == seq_len else v
                for k, v in base_inputs.items() if isinstance(v, torch.Tensor)
            }
            self.model.model(**prefix_fwd, past_key_values=cache)
            first_suffix_ids = base_inputs.input_ids[:, prefix_len:].to(device)
            del base_inputs, prefix_fwd

            # Generate each question's answer using cached prefix KV
            vision_end_marker = '<|vision_end|>'
            try:
                for i, line in enumerate(lines):
                    if line.get('index') in done_indices:
                        continue

                    if i == 0:
                        suffix_ids = first_suffix_ids
                    else:
                        text = self.processor.apply_chat_template(messages_list[i], tokenize=False, add_generation_prompt=True)
                        suffix_text = text[text.rfind(vision_end_marker) + len(vision_end_marker):]
                        suffix_ids = self.processor.tokenizer.encode(
                            suffix_text, return_tensors='pt', add_special_tokens=False
                        )

                    suffix_ids = suffix_ids.to(device)
                    suffix_len = suffix_ids.shape[1]
                    cache_position = torch.arange(prefix_len, prefix_len + suffix_len, device=device)
                    attention_mask = torch.ones(1, prefix_len + suffix_len, device=device)
                    generated_ids = self.model.generate(
                        input_ids=suffix_ids,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        cache_position=cache_position,
                        do_sample=False, max_new_tokens=16,
                        temperature=None, top_p=None, top_k=None,
                    )
                    cache.crop(prefix_len)
                    output_text = self.processor.decode(
                        generated_ids[0][suffix_ids.shape[1]:],
                        skip_special_tokens=True, clean_up_tokenization_spaces=False,
                    )

                    line['prediction'] = output_text
                    with open(self.result_file_path, 'a', encoding='utf-8') as f:
                        json.dump(line.to_dict() if hasattr(line, 'to_dict') else dict(line), f, ensure_ascii=False)
                        f.write('\n')
            finally:
                del cache, first_suffix_ids
                torch.cuda.empty_cache()

        logger.info(f'Inference finished, results saved to {self.result_file_path}')

    def evaluate(self):
        """Run evaluation"""
        eval_file = self.result_file_path
        if not osp.exists(eval_file):
            logger.error(f"Evaluation file not found: {eval_file}")
            return

        logger.info(f"Starting evaluation on {eval_file}")
        rating = self.VIDEO_DATASET_CLS.evaluate(eval_file)
        logger.info(json.dumps(rating, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run video evaluation with Qwen3-VL model')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--dataset', type=str, default='Video-MME')
    parser.add_argument('--output', type=str, default='./outputs/Qwen3-VL')
    parser.add_argument('--total_pixels', type=int, default=224000)
    parser.add_argument('--max_frames', type=int, default=2048)
    args = parser.parse_args()

    evaluator = Evaluator(
        model_path=args.model_path,
        dataset_name=args.dataset,
        output_dir=args.output,
        total_pixels=args.total_pixels,
        max_frames=args.max_frames,
    )
    evaluator.inference()
    evaluator.evaluate()
