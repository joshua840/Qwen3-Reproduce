import os
import os.path as osp
import logging

import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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
