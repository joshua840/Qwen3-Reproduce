"""
evaluate 흐름 파악용 테스트 코드
- dummy jsonl로 VideoMME, MLVU의 evaluate 과정을 단계별로 확인
- score 계산, rating 집계 과정의 input/output 추적
"""
import os
import sys
import json
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pandas as pd
from dataset.video_base import load_file, dump_file
from dataset.videomme import VideoMMEDataset, _extract_characters_regex, _get_dimension_rating as videomme_rating
from dataset.mlvu import MLVUDataset, _check_ans, _get_dimension_rating as mlvu_rating

INSPECT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_videomme_evaluate():
    print("=" * 60)
    print("VideoMME evaluate 흐름")
    print("=" * 60)

    # 1. load_file: jsonl → DataFrame
    eval_file = os.path.join(INSPECT_DIR, 'dummy_videomme.jsonl')
    data = load_file(eval_file)
    print(f"\n--- Step 1: load_file ---")
    print(f"input: {eval_file}")
    print(f"output: DataFrame, shape={data.shape}")
    print(f"columns: {list(data.columns)}")
    print(data[['index', 'answer', 'prediction']].to_string())

    # 2. _extract_characters_regex: prediction → A/B/C/D
    print(f"\n--- Step 2: _extract_characters_regex ---")
    for _, row in data.iterrows():
        pred = str(row['prediction'])
        extracted = _extract_characters_regex(pred)
        ans = row['answer']
        score = int(extracted == ans) if extracted else -1
        print(f"  idx={row['index']}: pred={pred!r:50s} → extracted={extracted!r:5s} vs ans={ans!r} → score={score}")

    # 3. score 컬럼 추가 (실제 evaluate 로직 재현)
    print(f"\n--- Step 3: score 계산 ---")
    for idx in data['index']:
        ans = data.loc[data['index'] == idx, 'answer'].values[0]
        pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])
        extracted = _extract_characters_regex(pred)
        data.loc[data['index'] == idx, 'score'] = int(extracted == ans) if extracted else -1

    rejected = [x for x in data['score'] if x == -1]
    print(f"total: {len(data)}, rejected (score=-1): {len(rejected)}")
    print(data[['index', 'answer', 'prediction', 'score']].to_string())

    # 4. dump_file → _score.tsv
    print(f"\n--- Step 4: dump_file → score.tsv ---")
    with tempfile.NamedTemporaryFile(suffix='_score.tsv', delete=False, dir=INSPECT_DIR) as f:
        score_file = f.name
    dump_file(data, score_file)
    reloaded = load_file(score_file)
    print(f"saved: {score_file}")
    print(f"reloaded shape: {reloaded.shape}, columns: {list(reloaded.columns)}")
    os.unlink(score_file)

    # 5. _get_dimension_rating
    print(f"\n--- Step 5: _get_dimension_rating ---")
    rating = videomme_rating(data)
    print(f"rating keys: {list(rating.keys())}")
    for duration in ['short', 'medium', 'long', 'overall']:
        print(f"\n  [{duration}]")
        print(f"    overall: {rating[duration]['overall']}")
        print(f"    domain: {rating[duration]['domain']}")
        print(f"    task_type: {rating[duration]['task_type']}")


def test_mlvu_evaluate():
    print("\n\n" + "=" * 60)
    print("MLVU evaluate 흐름")
    print("=" * 60)

    # 1. load_file
    eval_file = os.path.join(INSPECT_DIR, 'dummy_mlvu.jsonl')
    data = load_file(eval_file)
    print(f"\n--- Step 1: load_file ---")
    print(f"output: DataFrame, shape={data.shape}")
    print(f"columns: {list(data.columns)}")
    print(data[['index', 'answer', 'candidates', 'prediction']].to_string())

    # 2. _check_ans: prediction vs answer (with option format)
    print(f"\n--- Step 2: answer 변환 + _check_ans ---")
    for idx in data['index']:
        row = data.loc[data['index'] == idx].iloc[0]
        ans_raw = row['answer']
        pred = row['prediction']
        options = eval(row['candidates'])

        # answer를 "(A) Yellow" 형태로 변환
        answer_idx = -1
        for i, c in enumerate(options):
            if c == ans_raw:
                answer_idx = i
        ans_formatted = f"({chr(ord('A') + answer_idx)}) {ans_raw}"

        if 'Failed to obtain answer via API.' in str(pred):
            score = -1
            print(f"  idx={idx}: pred={pred!r:30s} → FAIL_MSG → score=-1")
        else:
            score = int(_check_ans(pred, ans_formatted))
            print(f"  idx={idx}: pred={pred!r:30s} vs ans={ans_formatted!r:25s} → score={score}")

    # 3. score 계산 (실제 evaluate 로직 재현)
    print(f"\n--- Step 3: score 계산 ---")
    FAIL_MSG = 'Failed to obtain answer via API.'
    for idx in data['index']:
        ans = data.loc[data['index'] == idx, 'answer'].values[0]
        pred = data.loc[data['index'] == idx, 'prediction'].values[0]
        options = eval(data.loc[data['index'] == idx, 'candidates'].values[0])
        answer_idx = -1
        for i, c in enumerate(options):
            if c == ans:
                answer_idx = i
        ans = f"({chr(ord('A') + answer_idx)}) {ans}"

        if FAIL_MSG in pred:
            data.loc[data['index'] == idx, 'score'] = -1
        else:
            data.loc[data['index'] == idx, 'score'] = int(_check_ans(pred, ans))

    rejected = [x for x in data['score'] if x == -1]
    print(f"total: {len(data)}, rejected (score=-1): {len(rejected)}")
    print(data[['index', 'task_type', 'answer', 'prediction', 'score']].to_string())

    # 4. _get_dimension_rating
    print(f"\n--- Step 4: _get_dimension_rating ---")
    rating = mlvu_rating(data)
    print(f"rating: task_type → [correct, total]")
    for task, (correct, total) in rating.items():
        print(f"  {task}: {correct}/{total} = {correct/total*100:.1f}%")


if __name__ == '__main__':
    test_videomme_evaluate()
    test_mlvu_evaluate()
