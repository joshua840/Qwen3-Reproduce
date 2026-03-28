import csv
import json
import os
import os.path as osp
import pickle
import warnings

import numpy as np
import pandas as pd


def LMUDataRoot():
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']
    home = osp.expanduser('~')
    root = osp.join(home, 'LMUData')
    os.makedirs(root, exist_ok=True)
    return root


def HFCacheRoot():
    cache_list = ['HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HOME']
    for cache_name in cache_list:
        if cache_name in os.environ and osp.exists(os.environ[cache_name]):
            if os.environ[cache_name].split('/')[-1] == 'hub':
                return os.environ[cache_name]
            else:
                return osp.join(os.environ[cache_name], 'hub')
    home = osp.expanduser('~')
    root = osp.join(home, '.cache', 'huggingface', 'hub')
    os.makedirs(root, exist_ok=True)
    return root


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,
                      (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    try:
        return handlers[suffix](data, f, **kwargs)
    except Exception:
        pkl_file = f.rsplit('.', 1)[0] + '.pkl'
        warnings.warn(f'Failed to dump to {suffix} format, falling back to pkl: {pkl_file}')
        return dump_pkl(data, pkl_file, **kwargs)


def get_pred_file_format():
    pred_format = os.getenv('PRED_FORMAT', '').lower()
    if pred_format == '':
        return 'xlsx'
    else:
        assert pred_format in ['tsv', 'xlsx', 'json'], f'Unsupported PRED_FORMAT {pred_format}'
        return pred_format


def get_eval_file_format():
    eval_format = os.getenv('EVAL_FORMAT', '').lower()
    if eval_format == '':
        return 'csv'
    else:
        assert eval_format in ['csv', 'json'], f'Unsupported EVAL_FORMAT {eval_format}'
        return eval_format


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def get_file_extension(file_path):
    return file_path.split('.')[-1]


def get_intermediate_file_path(eval_file, suffix, target_format=None):
    original_ext = get_file_extension(eval_file)

    def ends_with_list(s, lst):
        for item in lst:
            if s.endswith(item):
                return True
        return False

    if target_format is None:
        if ends_with_list(suffix, ['_tmp', '_response', '_processed']):
            target_format = 'pkl'
        elif ends_with_list(suffix, ['_rating', '_config', '_meta']):
            target_format = 'json'
        elif ends_with_list(suffix, ['_acc', '_fine', '_metrics']):
            target_format = get_eval_file_format()
        else:
            target_format = get_pred_file_format()

    return eval_file.replace(f'.{original_ext}', f'{suffix}.{target_format}')
