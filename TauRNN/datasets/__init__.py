'''Fetch datasets'''

def get_dataset(**kwargs):
  '''Function to return specified dataset'''
  if kwargs['name'] == 'simple':
    from .simple import SimpleDataset
    data = SimpleDataset(**kwargs)
    return data
  else:
    raise Exception(f'Dataset {kwargs["name"]} not recognised!')
