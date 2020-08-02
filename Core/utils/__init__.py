from .transforms import *
from .collate import *
from .arrange_data import *

def get_metrics(metrics):
    '''Function that returns a function definition for calculating training metrics'''
    if metrics == 'semantic_segmentation':
        from .metrics import metrics_semantic_segmentation
        return metrics_semantic_segmentation
    else:
        raise Exception(f'Metric function {metrics} does not exist!')

