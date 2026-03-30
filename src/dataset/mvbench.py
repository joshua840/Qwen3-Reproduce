import os
import os.path as osp

import pandas as pd

from .video_base import VideoDataset, get_cache_path


class MVBenchDataset(VideoDataset):

    def _default_data_root(self):
        root = get_cache_path('OpenGVLab/MVBench', branch='video')
        if root is not None:
            return root
        raise FileNotFoundError(
            'MVBench dataset not found in HF cache. Either:\n'
            '  huggingface-cli download OpenGVLab/MVBench --repo-type dataset --revision video\n'
            'Or provide data_root argument.'
        )

    def _load_data(self):
        df = pd.read_csv(osp.join(self.data_root, 'MVBench_MP4.tsv'), sep='\t')
        # Convert text answer to letter (answer column contains text like "Yellow")
        def text_to_letter(row):
            candidates = eval(row['candidates'])
            answer_idx = candidates.index(row['answer'])
            return chr(ord('A') + answer_idx)
        df['answer'] = df.apply(text_to_letter, axis=1)
        return df

    def _get_video_path(self, line):
        return os.path.join(self.data_root, line['prefix'], line['video'])

    @classmethod
    def _get_dimension_rating(cls, data):
        result_board = {}
        for _, item in data.iterrows():
            if item['task_type'] not in result_board:
                result_board[item['task_type']] = [0, 0]
            result_board[item['task_type']][1] += 1
            if item['score']:
                result_board[item['task_type']][0] += 1

        correct = 0
        total = 0
        for key, value in result_board.items():
            correct += value[0]
            total += value[1]
            result_board[key].append(f'{value[0] / value[1] * 100:.2f}%')

        result_board['overall'] = [correct, total, f'{correct / total * 100:.2f}%']
        return result_board
