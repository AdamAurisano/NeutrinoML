'''
Python module containing datasets
'''

def get_dataset(**kwargs):
  '''Function to return specified dataset'''
  if kwargs['name'] == 'SparsePixelMapNOvA':
    from .spm_nova import SparsePixelMapNOvA
    data = SparsePixelMapNOvA(**kwargs)
    return data
  else:
    raise Exception(f'Dataset {kwargs["name"]} not recognised!')
