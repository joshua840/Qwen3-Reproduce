import os
import os.path as osp
import re

import numpy as np

from .video_base import VideoDataset, get_cache_path

DURATIONS = ['short', 'medium', 'long']

DOMAINS = [
    'Knowledge', 'Film & Television', 'Sports Competition',
    'Artistic Performance', 'Life Record', 'Multilingual'
]

SUB_CATEGORIES = [
    'Humanity & History', 'Literature & Art', 'Biology & Medicine',
    'Finance & Commerce', 'Astronomy', 'Geography', 'Law', 'Life Tip',
    'Technology', 'Animation', 'Movie & TV Show', 'Documentary',
    'News Report', 'Esports', 'Basketball', 'Football', 'Athletics',
    'Other Sports', 'Stage Play', 'Magic Show', 'Variety Show',
    'Acrobatics', 'Handicraft', 'Food', 'Fashion', 'Daily Life',
    'Travel', 'Pet & Animal', 'Exercise', 'Multilingual'
]

TASK_CATEGORIES = [
    'Temporal Perception', 'Spatial Perception', 'Attribute Perception',
    'Action Recognition', 'Object Recognition', 'OCR Problems',
    'Counting Problem', 'Temporal Reasoning', 'Spatial Reasoning',
    'Action Reasoning', 'Object Reasoning', 'Information Synopsis',
]


class VideoMMEDataset(VideoDataset):

    def _default_data_root(self):
        root = get_cache_path('lmms-lab/Video-MME')
        if root is not None:
            return root
        raise FileNotFoundError(
            'Video-MME dataset not found in HF cache. Either:\n'
            '  huggingface-cli download lmms-lab/Video-MME --repo-type dataset\n'
            'Or provide data_root argument.'
        )

    def _load_data(self):
        from datasets import load_dataset
        df = load_dataset('lmms-lab/Video-MME', split='test').to_pandas()
        df['video'] = df['videoID']
        df['index'] = range(len(df))
        # Strip letter prefix from options: "A. Apples." -> "Apples."
        df['candidates'] = df['options'].apply(
            lambda opts: [re.sub(r'^[A-D]\.\s*', '', o) for o in opts]
        )
        return df

    def _get_video_path(self, line):
        return osp.join(self.data_root, 'video', f"{line['videoID']}.mp4")

    @classmethod
    def _get_dimension_rating(cls, data):
        duration_rating = {}
        for duration in DURATIONS + ['overall']:
            duration_rating[duration] = {
                'overall': '',
                'domain': {k: [] for k in DOMAINS},
                'sub_category': {k: [] for k in SUB_CATEGORIES},
                'task_type': {k: [] for k in TASK_CATEGORIES}
            }

        for i in range(len(data)):
            domain = data.iloc[i]['domain']
            sub_ctg = data.iloc[i]['sub_category']
            task_ctg = data.iloc[i]['task_type']
            duration = data.iloc[i]['duration']

            for d in [duration, 'overall']:
                duration_rating[d]['domain'][domain].append(data.iloc[i]['score'])
                duration_rating[d]['sub_category'][sub_ctg].append(data.iloc[i]['score'])
                duration_rating[d]['task_type'][task_ctg].append(data.iloc[i]['score'])

        for duration in DURATIONS + ['overall']:
            duration_rating[duration]['overall'] = f'{np.mean([x for x in sum(duration_rating[duration]["domain"].values(), []) if x >= 0]):.3f}'
            for domain in DOMAINS:
                duration_rating[duration]['domain'][domain] = f'{np.mean([x for x in duration_rating[duration]["domain"][domain] if x >= 0]):.3f}'
            for sub_ctg in SUB_CATEGORIES:
                duration_rating[duration]['sub_category'][sub_ctg] = f'{np.mean([x for x in duration_rating[duration]["sub_category"][sub_ctg] if x >= 0]):.3f}'
            for task_ctg in TASK_CATEGORIES:
                duration_rating[duration]['task_type'][task_ctg] = f'{np.mean([x for x in duration_rating[duration]["task_type"][task_ctg] if x >= 0]):.3f}'

        return duration_rating
