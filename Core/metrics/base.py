'''
Base class for training metric calculators
'''

from abc import ABC, abstractmethod

class MetricsBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def new_epoch(self):
        pass

    @abstractmethod
    def train_batch_metrics(self, y_pred, y_true):
        pass

    @abstractmethod
    def valid_batch_metrics(self, y_pred, y_true):
        pass

    @abstractmethod
    def epoch_metrics(self):
        pass

