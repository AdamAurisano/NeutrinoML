'''
Package for training metric classes
'''

def get_metrics(metrics):
    if metrics == 'SemanticSegmentation':
        from .semantic_segmentation import SemanticSegmentationMetrics
        return SemanticSegmentationMetrics
    elif metrics == 'Classification':
        from .classification import ClassificationMetrics
        return ClassificationMetrics
    else:
        raise Exception(f'Metric class "{metrics}" not recognised!')
