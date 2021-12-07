'''
Python module to contain graph datasets
'''

def get_dataset(**kwargs):
  '''Function to return specified dataset'''
  if kwargs['name'] == 'spgraph':
    from .spgraph import SPGraphDataset
    data = SPGraphDataset(**kwargs)
    return data
  elif kwargs['name'] == 'simple':
    from .simple import SimpleDataset
    data = SimpleDataset(**kwargs)
    return data
  else:
    raise Exception(f'Dataset {kwargs["name"]} not recognised!')

