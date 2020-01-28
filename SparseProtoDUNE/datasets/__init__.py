'''
Python module containing datasets
'''

def get_dataset(**kwargs):
    '''Function to return specified dataset'''
    # if kwargs['input_transform'] is not None:
    #  transform = getattr(torchvision.transforms, kwargs['input_transform'])
    if kwargs['name'] == 'SparsePixelMap':
        from .spm import SparsePixelMap
        data = SparsePixelMap(**kwargs)
        return data
    else:
        raise Exception(f'Dataset {kwargs["name"]} not recognised!')
