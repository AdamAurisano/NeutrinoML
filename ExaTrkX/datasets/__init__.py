'''
Python module to contain graph datasets
'''
from .norm import FeatureNorm

def get_dataset(**kwargs):
  '''Function to return specified dataset'''
  if kwargs['name'] == 'simple':
    from .simple import SimpleDataset
    data = SimpleDataset(**kwargs)
    return data
  elif kwargs['name'] == 'spacepoint':
    from .spacepoint import SpacePointDataset
    data = SpacePointDataset(**kwargs)
    return data
  else:
    raise Exception(f'Dataset {kwargs["name"]} not recognised!')

def training_dataset(**kwargs):
  kwargs["path"] = kwargs["path"] + "/train"
  return get_dataset(**kwargs)

def validation_dataset(**kwargs):
  kwargs["path"] = kwargs["path"] + "/valid"
  return get_dataset(**kwargs)

def test_dataset(**kwargs):
  kwargs["path"] = kwargs["path"] + "/test"
  return get_dataset(**kwargs)

