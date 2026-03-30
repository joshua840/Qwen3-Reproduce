import json
import os
import os.path as osp

import pandas as pd

from .video_base import VideoDataset, get_cache_path

# MCQ task JSONs in MLVU/MVLU repo (tasks 8, 9 are open-ended)
MCQ_JSONS = [
    '1_plotQA.json', '2_needle.json', '3_ego.json', '4_count.json',
    '5_order.json', '6_anomaly_reco.json', '7_topic_reasoning.json',
]


class MLVUDataset(VideoDataset):

    def _default_data_root(self):
        root = get_cache_path('MLVU/MVLU')
        if root is not None:
            mlvu = osp.join(root, 'MLVU')
            return mlvu if osp.isdir(mlvu) else root
        raise FileNotFoundError(
            'MLVU dataset not found in HF cache. Either:\n'
            '  huggingface-cli download MLVU/MVLU --repo-type dataset\n'
            'Or provide data_root argument.'
        )

    def _load_data(self):
        json_dir = osp.join(self.data_root, 'json')
        records = []
        for fname in MCQ_JSONS:
            task_dir = fname.replace('.json', '')  # e.g. "1_plotQA"
            items = json.load(open(osp.join(json_dir, fname), encoding='utf-8'))
            for item in items:
                # Convert text answer to letter
                answer_idx = item['candidates'].index(item['answer'])
                records.append({
                    'video': item['video'],
                    'video_dir': task_dir,
                    'duration': item['duration'],
                    'question': item['question'],
                    'candidates': item['candidates'],
                    'answer': chr(ord('A') + answer_idx),
                    'task_type': item.get('question_type', task_dir),
                })
        df = pd.DataFrame(records)
        df['index'] = range(len(df))
        return df

    def _get_video_path(self, line):
        return os.path.join(self.data_root, 'video', line['video_dir'], line['video'])

    @classmethod
    def _get_dimension_rating(cls, data):
        result_dict = {}
        for _, item in data.iterrows():
            if item['task_type'] not in result_dict:
                result_dict[item['task_type']] = [0, 0]
            result_dict[item['task_type']][0] += int(item['score'])
            result_dict[item['task_type']][1] += 1
        return result_dict
