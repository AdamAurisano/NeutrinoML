from .transforms import *
from .collate import *
#from .arrange_data import *

def get_metrics(metrics):
    '''Function that returns a function definition for calculating training metrics'''
    if metrics == 'semantic_segmentation':
        from .metrics import metrics_semantic_segmentation
        return metrics_semantic_segmentation
    else:
        raise Exception(f'Metric function {metrics} does not exist!')
def get_arrange(arrange_data):
    '''Function that returns a function to arrange de format of data depending on Sparse or Minkowski Network '''
    if arrange_data == 'arrange_sparse':
        from .arrange_data import arrange_sparse
        return arrange_sparse
    if arrange_data == 'arrange_sparse_minkowski':
        from .arrange_data import arrange_sparse_minkowski
        return arrange_sparse_minkowski
    if arrange_data == 'arrange_sparse_minkowski_2stack':
        from .arrange_data import arrange_sparse_minkowski_2stack
        return arrange_sparse_minkowski_2stack
    else:
        raise Exception(f'Function {arrange_data} does not exist!')
