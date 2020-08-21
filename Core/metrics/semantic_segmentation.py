'''
Class for calculating training metrics for a semantic segmentation network
'''

from .base import MetricsBase
import psutil

class SemanticSegmentationMetrics(MetricsBase):
    '''Class for calculating training metrics for a semantic segmentation network'''
    def __init__(self, class_names):
        self.class_names = class_names
        self.new_epoch()

    @property
    def n_classes(self): return len(self.class_names)

    def new_epoch(self):
        '''Reset metrics at the start of a new epoch'''

        # Overall training metrics
        self.train_correct = 0
        self.train_total   = 0
        self.valid_correct = 0
        self.valid_total   = 0

        # Class-wise training metrics
        self.train_class_correct = [ 0 for i in range(self.n_classes) ]
        self.train_class_total   = [ 0 for i in range(self.n_classes) ]
        self.valid_class_correct = [ 0 for i in range(self.n_classes) ]
        self.valid_class_total   = [ 0 for i in range(self.n_classes) ]

    def __accuracy_helper(self, y_pred, y_true):
        '''Utility function to help calculate accuracy for a batch'''

        w_pred = y_pred.argmax(dim=0)
        w_true = y_true.argmax(dim=0)
        correct = (w_pred==w_true)

        # Calculate batch accuracy
        batch_correct = correct.sum().float().item()
        batch_total   = w_pred.shape[0]

        # Calculate accuracy for each class
        class_correct = [ (w_pred[correct]==i).sum().float().item() for i in range(self.n_classes) ]
        class_total = [ (w_true==i).sum().float().item() for i in range(self.n_classes) ]
        return (batch_correct, batch_total, class_correct, class_total)

    def train_batch_metrics(self, y_pred, y_true):
        '''Function to calculate metrics for each training batch'''

        # Sanity check
        if len(self.class_names) != y_pred.shape[1]:
            raise Exception(f'Number of class names ({len(class_names)}) does not match shape of network output ({y_pred.shape[1]})!')

        metrics = {}

        batch_correct, batch_total, class_correct, class_total = self.__accuracy_helper(y_pred, y_true)

        metrics['acc/batch'] = 100 * batch_correct / batch_total
        self.train_correct += batch_correct
        self.train_total += batch_total

        for i, name in enumerate(self.class_names):
            metrics[f'acc_class/batch/{name}'] = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            self.train_class_correct[i] += class_correct[i]
            self.train_class_total[i] += class_total[i]

        metrics['memory'] = psutil.virtual_memory().used

        return metrics

    def valid_batch_metrics(self, y_pred, y_true):
        '''Function to calculate metrics for each validation batch'''

        # Sanity check
        if len(self.class_names) != y_pred.shape[1]:
            raise Exception(f'Number of class names ({len(class_names)}) does not match shape of network output ({y_pred.shape[1]})!')

        batch_correct, batch_total, class_correct, class_total = self.__accuracy_helper(y_pred, y_true)
        self.valid_correct += batch_correct
        self.valid_total += batch_total
        for i, name in enumerate(self.class_names):
            self.valid_class_correct[i] += class_correct[i]
            self.valid_class_total[i] += class_total[i]

    def epoch_metrics(self):
        '''Function to calculate metrics for each epoch'''

        metrics = {}
        metrics['acc/epoch'] = {
            'train': 100 * self.train_correct / self.train_total,
            'valid': 100 * self.valid_correct / self.valid_total
        }
        for i, name in enumerate(self.class_names):
            metrics[f'acc_class/epoch/{name}'] = {
                'train': 100 * self.train_class_correct[i] / self.train_class_total[i] if self.train_class_total[i] > 0 else 0,
                'valid': 100 * self.valid_class_correct[i] / self.valid_class_total[i] if self.valid_class_total[i] > 0 else 0
            }
        return metrics

