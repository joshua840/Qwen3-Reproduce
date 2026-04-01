import json
import os.path as osp

import pandas as pd

from .video_base import VideoDataset, get_cache_path


class LongVideoBenchDataset(VideoDataset):

    MCQ_INSTRUCTION = (
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter of the correct option."
    )

    def _default_data_root(self):
        root = get_cache_path('longvideobench/LongVideoBench')
        if root is not None:
            return root
        raise FileNotFoundError(
            'LongVideoBench dataset not found in HF cache. Either:\n'
            '  huggingface-cli download longvideobench/LongVideoBench --repo-type dataset\n'
            'Or provide data_root argument.'
        )

    def _load_data(self):
        json_path = osp.join(self.data_root, 'lvb_val.json')
        items = json.load(open(json_path, encoding='utf-8'))
        records = []
        for item in items:
            candidates = item['candidates']
            records.append({
                'video': item['video_id'],
                'question': item['question'],
                'candidates': candidates,
                'answer': chr(ord('A') + item['correct_choice']),
                'topic_category': item['topic_category'],
                'question_category': item['question_category'],
                'level': item['level'],
                'duration_group': item['duration_group'],
                'video_path': item['video_path'],
            })
        df = pd.DataFrame(records)
        df['index'] = range(len(df))
        return df

    def _get_video_path(self, line):
        return osp.join(self.data_root, 'videos', line['video_path'])

    @classmethod
    def _get_dimension_rating(cls, data):
        result = {}
        for key in ['level', 'topic_category', 'question_category', 'duration_group']:
            group_dict = {}
            for _, item in data.iterrows():
                g = str(item[key])
                if g not in group_dict:
                    group_dict[g] = [0, 0]
                group_dict[g][0] += int(item['score'])
                group_dict[g][1] += 1
            result[key] = group_dict
        return result
