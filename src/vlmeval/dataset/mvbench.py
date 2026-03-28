import os
import os.path as osp

import pandas as pd

from vlmeval.smp import dump, get_file_extension, get_intermediate_file_path, load
from .video_base import VideoDataset

FAIL_MSG = 'Failed to obtain answer via API.'


def _check_ans(pred, gt):
    pred_option = pred.lower().strip().split(' ')[0]
    gt_option = gt.lower().strip().split(' ')[0]
    if pred_option.replace('.', '') in gt_option:
        return True
    if gt_option in pred_option:
        return True
    return False


def _get_dimension_rating(data_path):
    data = load(data_path)
    result_board = {}
    for idx, item in data.iterrows():
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


class MVBenchDataset(VideoDataset):

    SYS = """Carefully watch the video and pay attention to the cause and sequence of events, \
the detail and movement of objects, and the action and pose of persons. \
Based on your observations, select the best option that accurately addresses the question.
"""

    def __init__(self, **kwargs):
        from vlmeval.smp import get_cache_path
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
        struct = [dict(type='text', value=self.SYS, role='system')]
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])
        struct.append(dict(type='video', value=video_path))
        struct.append(dict(type='text', value=question))
        struct.append(dict(type='text', value='\nOnly give the best option.'))
        struct.append(dict(type='text', value='Best option:(', role='assistant'))
        return struct

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv', 'jsonl']

        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data_un['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]
                options = eval(data.loc[data['index'] == idx, 'candidates'].values[0])
                answer_idx = -1
                for id, c in enumerate(options):
                    if c == ans:
                        answer_idx = id
                ans = f"({chr(ord('A') + answer_idx)}) {ans}"

                if FAIL_MSG in pred:
                    data.loc[idx, 'score'] = -1
                else:
                    data.loc[idx, 'score'] = int(_check_ans(pred, ans))

            rejected = [x for x in data['score'] if x == -1]
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to extract answer for another {len(rejected)} questions. '
            )

            dump(data, score_file)

        rating = _get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
