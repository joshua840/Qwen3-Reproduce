import os
import sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import argparse
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, DynamicCache, DynamicCache
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from vlmeval.smp import *
from vlmeval.vlm.qwen3_vl.model import ensure_video_url
from vlmeval.dataset.videomme import VideoMME
from vlmeval.dataset.mvbench import MVBench_MP4
from vlmeval.dataset.mlvu import MLVU_MCQ

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VideoDataset(Dataset):
    """Dataset grouped by video. Each item = one video with all its questions."""

    def __init__(self, dataset_name=None, data_root=None, data_path=None, system_prompt=None,
                 total_pixels=224000, max_frames=2048):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.data_path = data_path
        self.total_pixels = total_pixels
        self.max_frames = max_frames
        logger.info(f'Loading data from {self.data_path}')
        self.data = pd.read_csv(self.data_path, sep='\t')
        self.system_prompt = system_prompt

        self.groups = [group for _, group in self.data.groupby('video', sort=False)]
        logger.info(f'Loaded {len(self.data)} questions across {len(self.groups)} videos')

    def __len__(self):
        return len(self.groups)

    def _build_struct(self, line):
        raise NotImplementedError

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError

    def _prepare_content(self, inputs):
        content = []
        for s in inputs:
            if s['type'] == 'video':
                content.append({
                    'type': 'video',
                    'video': ensure_video_url(s['value']),
                    'min_pixels': 128 * 32 * 32,
                    'max_pixels': 768 * 32 * 32,
                    'total_pixels': self.total_pixels * 32 * 32,
                    'max_frames': self.max_frames,
                    'fps': 2,
                })
            elif s['type'] == 'text':
                content.append({'type': 'text', 'text': s['value']})
        return content

    def _build_messages(self, line):
        struct = self._build_struct(line)
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(struct)})
        return messages

    def __getitem__(self, index):
        """Returns (lines, messages_list) — raw data only, no model processing."""
        group = self.groups[index]
        lines = []
        messages_list = []
        for _, line in group.iterrows():
            lines.append(line)
            messages_list.append(self._build_messages(line))
        return lines, messages_list
    
class VideoMMEDataset(VideoDataset):
    def __init__(self, **kwargs):
        data_root = get_cache_path('lmms-lab/Video-MME')
        assert data_root is not None, (
            'Video-MME dataset not found in HF cache. Run:\n'
            '  huggingface-cli download lmms-lab/Video-MME --repo-type dataset'
        )
        data_path = osp.join(data_root, 'Video-MME.tsv')
        super().__init__(dataset_name='Video-MME', data_root=data_root, data_path=data_path, **kwargs)

    def _build_struct(self, line):
        struct = []
        video_path = osp.join(self.data_root, line['video_path'])
        struct.append(dict(type='video', value=video_path))
        struct.append(dict(type='text', value=VideoMME.FRAMES_TMPL_NOSUB))
        question = line['question'] + '\n' + '\n'.join(eval(line['candidates']))
        struct.append(dict(type='text', value=f'Question: {question}\nAnswer: '))
        return struct

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        judge_kwargs = dict(
            model='qwen__qwen3-vl-235b-a22b-instruct',
            nproc=4
        )
        return VideoMME.evaluate(eval_file, **judge_kwargs)
    
class MVBenchDataset(VideoDataset):
    def __init__(self, **kwargs):
        data_root = get_cache_path('OpenGVLab/MVBench', branch='video')
        assert data_root is not None, (
            'MVBench dataset not found in HF cache. Run:\n'
            '  huggingface-cli download OpenGVLab/MVBench --repo-type dataset --revision video'
        )
        data_path = osp.join(data_root, 'MVBench_MP4.tsv')
        super().__init__(dataset_name='MVBench_MP4', data_root=data_root, data_path=data_path, **kwargs)

    def _qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += 'Options:\n'
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(eval(data['candidates'])):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer
    
    def _build_struct(self, line):
        question, answer = self._qa_template(line)
        struct = [dict(type='text', value=MVBench_MP4.SYS, role='system')]
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])
        struct.append(dict(type='video', value=video_path))
        struct.append(dict(type='text', value=question))
        struct.append(dict(type='text', value='\nOnly give the best option.'))
        struct.append(dict(type='text', value='Best option:(', role='assistant'))
        return struct
    
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        return MVBench_MP4.evaluate(eval_file, **judge_kwargs)

class MLVUDataset(VideoDataset):
    SYS = MLVU_MCQ.BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'

    def __init__(self, **kwargs):
        data_root = '/data/MLVU'
        data_path = osp.join(data_root, 'MLVU_MCQ.tsv')
        assert osp.exists(data_path), f'MLVU TSV not found: {data_path}'
        super().__init__(dataset_name='MLVU_MCQ', data_root=data_root, data_path=data_path, **kwargs)

    def _qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += 'Options:\n'
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(eval(data['candidates'])):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def _build_struct(self, line):
        question, answer = self._qa_template(line)
        struct = [dict(type='text', value=self.SYS, role='system')]
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])
        struct.append(dict(type='video', value=video_path))
        struct.append(dict(type='text', value=question))
        struct.append(dict(type='text', value='\nOnly give the best option.'))
        return struct

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        return MLVU_MCQ.evaluate(eval_file, **judge_kwargs)

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

        DATASET_MAP = {
            'Video-MME': VideoMMEDataset,
            'MVBench': MVBenchDataset,
            'MLVU': MLVUDataset,
        }
        if dataset_name not in DATASET_MAP:
            raise ValueError(f'Unsupported dataset: {dataset_name}. Choose from {list(DATASET_MAP.keys())}')
        self.VIDEO_DATASET_CLS = DATASET_MAP[dataset_name]

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.dataset = self.VIDEO_DATASET_CLS(total_pixels=total_pixels, max_frames=max_frames)
        logger.info(f'Initializing model from {self.model_path}')
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

        # Convert jsonl → tsv for the evaluate method which expects tsv/xlsx/json
        tsv_file = eval_file.replace('.jsonl', '.tsv')
        records = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        df.to_csv(tsv_file, sep='\t', index=False)
        logger.info(f"Starting evaluation on {tsv_file} ({len(df)} samples)")

        rating = self.VIDEO_DATASET_CLS.evaluate(tsv_file)
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