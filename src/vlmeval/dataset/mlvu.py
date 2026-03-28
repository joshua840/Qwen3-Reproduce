import os
import os.path as osp

import pandas as pd

from .video_base import VideoDataset, load_file, dump_file

FAIL_MSG = 'Failed to obtain answer via API.'


def _check_ans(pred, gt):
    index = gt.index("(")
    index2 = gt.index(")")
    gt_option = gt[index + 1: index2]

    if ")" in pred:
        index3 = pred.index(")")
        pred = pred[index3 - 1: index3]
    return pred == gt_option


def _get_dimension_rating(data):
    result_dict = {}
    for idx, item in data.iterrows():
        if item['task_type'] not in result_dict:
            result_dict[item['task_type']] = [0, 0]
        result_dict[item['task_type']][0] += int(item['score'])
        result_dict[item['task_type']][1] += 1
    return result_dict


class MLVUDataset(VideoDataset):

    BASE_SYS = 'Carefully watch this video and pay attention to every detail. '
    SYS = BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'

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
    def evaluate(cls, eval_file, **kwargs):
        score_file = eval_file.rsplit('.', 1)[0] + '_score.tsv'

        if not osp.exists(score_file):
            data = load_file(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
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

            dump_file(data, score_file)

        scored_data = load_file(score_file)
        rating = _get_dimension_rating(scored_data)
        return rating
