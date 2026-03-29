import json
import os
import os.path as osp
import logging
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
    """Load a file by extension (tsv/json/jsonl/parquet)."""
    suffix = f.split('.')[-1]
    if suffix == 'tsv':
        return pd.read_csv(f, sep='\t')
    elif suffix == 'json':
        return json.load(open(f, 'r', encoding='utf-8'))
    elif suffix == 'jsonl':
        lines = open(f, encoding='utf-8').readlines()
        return pd.DataFrame([json.loads(x.strip()) for x in lines if x.strip()])
    elif suffix == 'csv':
        return pd.read_csv(f)
    elif suffix == 'parquet':
        return pd.read_parquet(f)
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
        self.data = load_file(self.data_path)
        self._transform_data()
        self.system_prompt = system_prompt

        self.groups = [group for _, group in self.data.groupby('video', sort=False)]
        logger.info(f'Loaded {len(self.data)} questions across {len(self.groups)} videos')

    def __len__(self):
        return len(self.groups)

    def _transform_data(self):
        """Override to transform raw data after loading (e.g. column mapping)."""
        pass

    def _build_struct(self, line):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        raise NotImplementedError

    def _prepare_content(self, inputs):
        content = []
        for s in inputs:
            if s['type'] == 'video':
                content.append({
                    'type': 'video',
                    'video': s['value'],
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
