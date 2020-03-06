'''
Python module containing datasets
'''

def get_dataset(**kwargs):
  '''Function to return specified dataset'''
  # if kwargs['input_transform'] is not None:
  #  transform = getattr(torchvision.transforms, kwargs['input_transform'])
  if kwargs['name'] == 'SparsePixelMap2D':
    from .SPM import SparsePixelMap
    data = SparsePixelMap(**kwargs)
    return data
  if kwargs['name'] == 'SparsePixelMap3D':
    from .SPM_3D import SparsePixelMap3D
    data = SparsePixelMap3D(**kwargs)
    return data
  else:
    raise Exception(f'Dataset {kwargs["name"]} not recognised!')
