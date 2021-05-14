"""
Class for calculating training metrics for a graph network
"""

from .base import MetricsBase
import time, psutil, torch, torch.cuda as tc
from torch.nn.functional import softmax

class GraphMetrics(MetricsBase):
  """Class for calculating training metrics for a graph network"""
  def __init__(self, class_names):
    self.class_names = class_names
    self.new_epoch()

  @property
  def n_classes(self): return len(self.class_names)

  def new_epoch(self):
    """Reset metrics at the start of a new epoch"""

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

    self.epoch_start = time.time()

  def __accuracy_helper(self, y_pred, y_true):
    """Utility function to help calculate accuracy for a batch"""

    w_pred = softmax(y_pred, dim=1).argmax(dim=1).cpu().numpy()
    w_true = y_true.cpu().numpy()
    correct = (w_pred==w_true)

    # Calculate batch accuracy
    batch_correct = float(correct.sum())
    batch_total   = float(w_pred.shape[0])

    # Calculate accuracy for each class
    class_correct = [ float((w_pred[correct]==i).sum()) for i in range(self.n_classes) ]
    class_total = [ float((w_true==i).sum()) for i in range(self.n_classes) ]
    return (batch_correct, batch_total, class_correct, class_total)

  def train_batch_metrics(self, y_pred, y_true):
    """Function to calculate metrics for each training batch"""

    with torch.no_grad():

      #B, C = y_pred
      #y_pred = torch.cat([B.unsqueeze(dim=-1), C], dim=1)

      # Sanity check
      if len(self.class_names) != y_pred.shape[1]:
          raise Exception(f"Number of class names ({len(self.class_names)}) does not match shape of network output ({y_pred.shape[1]})!")

      metrics = {}

      batch_correct, batch_total, class_correct, class_total = self.__accuracy_helper(y_pred, y_true)
      metrics["acc/batch"] = 100 * batch_correct / batch_total
      self.train_correct += batch_correct
      self.train_total += batch_total

      for i, name in enumerate(self.class_names):
        metrics[f"acc_class/batch/{name}"] = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        self.train_class_correct[i] += class_correct[i]
        self.train_class_total[i] += class_total[i]

      metrics["batch_hits"] = y_pred.shape[0]

      metrics["memory/cpu"] = float(psutil.virtual_memory().used) / float(1073741824)
      metrics["memory/gpu"] = float(tc.memory_reserved(y_pred.device)) / float(1073741824)
      metrics["memory/gpu_max"] = float(tc.max_memory_allocated(y_pred.device)) / float(1073741824)

      self.train_end = time.time()

      return metrics

  def valid_batch_metrics(self, y_pred, y_true):
    """Function to calculate metrics for each validation batch"""

    with torch.no_grad():

      #B, C = y_pred
      #y_pred = torch.cat([B.unsqueeze(dim=-1), C], dim=1)

      # Sanity check
      if len(self.class_names) != y_pred.shape[1]:
        raise Exception(f"Number of class names ({len(class_names)}) does not match shape of network output ({y_pred.shape[1]})!")

      batch_correct, batch_total, class_correct, class_total = self.__accuracy_helper(y_pred, y_true)
      self.valid_correct += batch_correct
      self.valid_total += batch_total
      for i, name in enumerate(self.class_names):
        self.valid_class_correct[i] += class_correct[i]
        self.valid_class_total[i] += class_total[i]

      self.valid_end = time.time()

  def epoch_metrics(self):
    """Function to calculate metrics for each epoch"""

    with torch.no_grad():

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
      metrics['time/epoch'] = {
        'train': self.train_end - self.epoch_start,
        'valid': self.valid_end - self.train_end
      }
      return metrics

