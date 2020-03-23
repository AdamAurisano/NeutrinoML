'''
Python module to contain graph datasets
'''
#import torchvision

def get_dataset(**kwargs):
  '''Function to return specified dataset'''
 # if kwargs['input_transform'] is not None:
  #  transform = getattr(torchvision.transforms, kwargs['input_transform'])
  if kwargs['name'] == 'spgraph':
    from .spgraph import SPGraphDataset
    data = SPGraphDataset(**kwargs)
    return data
  else:
    raise Exception(f'Dataset {kwargs["name"]} not recognised!')

