from .transforms import *
from .collate import *
from Core.utils import arrange_data, arrange_truth
# import arrange_data, arrange_truth

def get_arrange_data(name):
    '''Function that returns a function to arrange input tensors across a batch'''
    ret = getattr(arrange_data, name)
    if ret is None: raise Exception(f'Function {name} does not exist!')
    else: return ret

def get_arrange_truth(name):
    '''Function that returns a function to arrange ground truth across a batch'''
    ret = getattr(arrange_truth, name)
    if ret is None: raise Exception(f'Function {name} does not exist!')
    else: return ret
    