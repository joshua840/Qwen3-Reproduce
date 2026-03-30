# from huggingface_hub import repo_info

# info = repo_info('MLVU/MLVU', repo_type='dataset')
# print(f'ID: {info.id}')
# print(f'Files:')
# for s in info.siblings[:20]:
#     print(f'  {s.rfilename}')


from datasets import load_dataset, get_dataset_split_names

print(get_dataset_split_names('MLVU/MVLU'))
ds = load_dataset('MLVU/MVLU', split='test')
print(ds.column_names, len(ds))
print(ds[0])