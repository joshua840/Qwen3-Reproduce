import json
import os
import os.path as osp
import logging
import re
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def get_cache_path(repo_id, branch='main', repo_type='datasets'):
    """Find the local snapshot path of a HuggingFace cached dataset."""
    try:
        from huggingface_hub.utils._cache_manager import _scan_cached_repo
        hf_home = os.environ.get('HF_HOME', osp.join(osp.expanduser('~'), '.cache', 'huggingface'))
        cache_path = os.environ.get('HF_HUB_CACHE', osp.join(hf_home, 'hub'))
        org, repo_name = repo_id.split('/')
        repo_path = Path(osp.join(cache_path, f'{repo_type}--{org}--{repo_name}/'))
        hf_cache_info = _scan_cached_repo(repo_path=repo_path)
        revs = {r.refs: r for r in hf_cache_info.revisions}
        if branch is not None:
            revs = {refs: r for refs, r in revs.items() if branch in refs}
        rev2keep = max(revs.values(), key=lambda r: r.last_modified)
        return str(rev2keep.snapshot_path)
    except Exception as e:
        logging.warning(f'{type(e)}: {e}')
        return None


def load_file(f):
    """Load a file by extension (tsv/json/jsonl)."""
    suffix = f.split('.')[-1]
    if suffix == 'tsv':
        return pd.read_csv(f, sep='\t')
    elif suffix == 'json':
        data = json.load(open(f, 'r', encoding='utf-8'))
        if isinstance(data, list):
            return pd.DataFrame(data)
        return data
    elif suffix == 'jsonl':
        lines = open(f, encoding='utf-8').readlines()
        return pd.DataFrame([json.loads(x.strip()) for x in lines if x.strip()])
    elif suffix == 'csv':
        return pd.read_csv(f)
    else:
        raise ValueError(f'Unsupported file format: {suffix}')


def dump_file(data, f):
    """Dump data to a file by extension (tsv/json)."""
    suffix = f.split('.')[-1]
    if suffix == 'tsv':
        data.to_csv(f, sep='\t', index=False)
    elif suffix == 'json':
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        json.dump(data, open(f, 'w'), indent=4, ensure_ascii=False)
    elif suffix == 'csv':
        data.to_csv(f, index=False)
    else:
        raise ValueError(f'Unsupported file format: {suffix}')


def extract_letter(s):
    """Extract the first A/B/C/D letter from a model prediction string."""
    s = s.strip()
    for prefix in [
        'The best answer is', 'The correct answer is', 'The answer is',
        'The answer', 'The best option is', 'The correct option is',
        'Best answer:', 'Best option:', 'Answer:', 'Option:',
    ]:
        s = s.replace(prefix, '')
    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    return matches[0] if matches else ''


class VideoDataset(Dataset):
    """Dataset grouped by video. Each item = one video with all its questions."""

    MCQ_INSTRUCTION = (
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )

    def __init__(self, data_root=None, total_pixels=224000, max_frames=2048,
                 patch_size=16, min_frame_tokens=128, max_frame_tokens=768):
        if data_root is None:
            data_root = self._default_data_root()
        self.data_root = data_root
        self.total_pixels = total_pixels
        self.max_frames = max_frames
        self.patch_size = patch_size
        self.min_frame_tokens = min_frame_tokens
        self.max_frame_tokens = max_frame_tokens
        self.data = self._load_data() # metadata
        self.groups = [group for _, group in self.data.groupby('video', sort=False)]
        logger.info(f'Loaded {len(self.data)} questions across {len(self.groups)} videos')

    def __len__(self):
        return len(self.groups)

    def _default_data_root(self):
        raise NotImplementedError

    def _load_data(self):
        raise NotImplementedError

    def _get_video_path(self, line):
        raise NotImplementedError

    def __getitem__(self, index):
        """Returns (lines, messages_list) — raw data only, no model processing."""
        group = self.groups[index]
        lines = []
        messages_list = []
        for _, line in group.iterrows():
            lines.append(line)
            messages_list.append(self._build_messages(line))
        return lines, messages_list
    
    def _build_messages(self, line):
        candidates = line['candidates']
        if isinstance(candidates, str):
            candidates = eval(candidates)
        options = '\n'.join(f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(candidates))
        prompt = (
            f"{self.MCQ_INSTRUCTION}\n"
            f"Question: {line['question']} Possible answer choices:\n"
            f"{options}\n"
            f"The best answer is:"
        )
        pf2 = (2 * self.patch_size) ** 2
        content = [
            {
                'type': 'video',
                'video': self._get_video_path(line),
                'min_pixels': self.min_frame_tokens * pf2,
                'max_pixels': self.max_frame_tokens * pf2,
                'total_pixels': self.total_pixels * pf2,
                'max_frames': self.max_frames,
                'fps': 2,
            },
            {'type': 'text', 'text': prompt},
        ]
        return [{'role': 'user', 'content': content}]

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        scored_data, _ = cls._score_predictions(eval_file)
        rating = cls._get_dimension_rating(scored_data)
        eval_dir = osp.dirname(eval_file)
        dump_file(rating, osp.join(eval_dir, 'rating.json'))
        return rating

    @classmethod
    def _score_predictions(cls, eval_file):
        """Score predictions by extracting letters and comparing to ground truth.
        Returns (scored_data, score_file_path). Skips if score file already exists."""
        score_file = osp.join(osp.dirname(eval_file), 'score.tsv')

        if not osp.exists(score_file):
            data = load_file(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
                extracted = extract_letter(pred)
                data.loc[data['index'] == idx, 'score'] = int(extracted == ans) if extracted else -1

            rejected = [x for x in data['score'] if x == -1]
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to extract answer for another {len(rejected)} questions. '
            )
            dump_file(data, score_file)

        return load_file(score_file), score_file

    @classmethod
    def _get_dimension_rating(cls, data):
        raise NotImplementedError