import os.path as osp
import re

import numpy as np
import pandas as pd

from .video_base import VideoDataset, get_cache_path, load_file, dump_file

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


def _extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is', 'The correct answer is', 'The answer is',
        'The answer', 'The best option is', 'The correct option is',
        'Best answer:', 'Best option:', 'Answer:', 'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]


def _get_dimension_rating(data):
    duration_rating = {k: {} for k in DURATIONS}
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

        duration_rating[duration]['domain'][domain].append(data.iloc[i]['score'])
        duration_rating[duration]['sub_category'][sub_ctg].append(data.iloc[i]['score'])
        duration_rating[duration]['task_type'][task_ctg].append(data.iloc[i]['score'])

        duration_rating['overall']['domain'][domain].append(data.iloc[i]['score'])
        duration_rating['overall']['sub_category'][sub_ctg].append(data.iloc[i]['score'])
        duration_rating['overall']['task_type'][task_ctg].append(data.iloc[i]['score'])

    for duration in DURATIONS + ['overall']:
        overall_res_dur = f'{np.mean([x for x in sum(duration_rating[duration]["domain"].values(), []) if x >= 0]):.3f}'
        duration_rating[duration]['overall'] = overall_res_dur

        for domain in DOMAINS:
            domain_res_dur = f'{np.mean([x for x in duration_rating[duration]["domain"][domain] if x >= 0]):.3f}'
            duration_rating[duration]['domain'][domain] = domain_res_dur

        for sub_ctg in SUB_CATEGORIES:
            sub_res_dur = f'{np.mean([x for x in duration_rating[duration]["sub_category"][sub_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['sub_category'][sub_ctg] = sub_res_dur

        for task_ctg in TASK_CATEGORIES:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_type"][task_ctg] if x >= 0]):.3f}'
            duration_rating[duration]['task_type'][task_ctg] = task_res_dur

    return duration_rating


class VideoMMEDataset(VideoDataset):

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

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
        struct.append(dict(type='text', value=self.FRAMES_TMPL_NOSUB))
        question = line['question'] + '\n' + '\n'.join(eval(line['candidates']))
        struct.append(dict(type='text', value=f'Question: {question}\nAnswer: '))
        return struct

    @classmethod
    def evaluate(cls, eval_file, **kwargs):
        score_file = eval_file.rsplit('.', 1)[0] + '_score.tsv'
        rating_file = eval_file.rsplit('.', 1)[0] + '_rating.json'

        if not osp.exists(score_file):
            data = load_file(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
                extracted = _extract_characters_regex(pred)
                data.loc[data['index'] == idx, 'score'] = int(extracted == ans) if extracted else -1

            rejected = [x for x in data['score'] if x == -1]
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to extract answer for another {len(rejected)} questions. '
            )

            dump_file(data, score_file)

        scored_data = load_file(score_file)
        rating = _get_dimension_rating(scored_data)
        dump_file(rating, rating_file)
        return rating
