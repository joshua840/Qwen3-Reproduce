import os
import os.path as osp
from pathlib import Path

from huggingface_hub.utils._cache_manager import _scan_cached_repo


def modelscope_flag_set():
    return os.environ.get('VLMEVALKIT_USE_MODELSCOPE', None) in ['1', 'True']


def get_cache_path(repo_id, branch='main', repo_type='datasets'):
    try:
        if modelscope_flag_set():
            from modelscope.hub.file_download import create_temporary_directory_and_cache
            if repo_type == 'datasets':
                repo_type = 'dataset'
            _, cache = create_temporary_directory_and_cache(model_id=repo_id, repo_type=repo_type)
            cache_path = cache.get_root_location()
            return cache_path
        else:
            from .file import HFCacheRoot
            cache_path = HFCacheRoot()
            org, repo_name = repo_id.split('/')
            repo_path = Path(osp.join(cache_path, f'{repo_type}--{org}--{repo_name}/'))
            hf_cache_info = _scan_cached_repo(repo_path=repo_path)
            revs = {r.refs: r for r in hf_cache_info.revisions}
            if branch is not None:
                revs = {refs: r for refs, r in revs.items() if branch in refs}
            rev2keep = max(revs.values(), key=lambda r: r.last_modified)
            return str(rev2keep.snapshot_path)
    except Exception as e:
        import logging
        logging.warning(f'{type(e)}: {e}')
        return None
