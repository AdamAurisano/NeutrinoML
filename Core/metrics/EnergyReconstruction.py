from .base import MetricsBase
import time, psutil, torch.cuda as tc
import torch

class EnergyReconstructionMetrics(MetricsBase):
    def __init__(self, class_names):
        self.class_names = class_names
        self.new_epoch()
        
    def new_epoch(self):
        '''Reset metrics at the start of a new epoch'''
        self.train_mean = None
        #self.train_std = None
        self.valid_mean = None
        #self.valid_std = None


    def train_batch_metrics(self, output, target):
        metrics = {}
        d = target-output
        d_mean = torch.mean(d)[None] 
        d_std  = torch.std(d)
        #d = self.EnergyRegression(output, target)
        if self.train_mean is None:
            self.train_mean = d_mean  
        else:
            self.train_mean = d #torch.cat((self.train_mean,d_mean), dim=0)

        metrics['mean/batch'] = d_mean
        metrics['std/batch'] = d_std

        metrics['memory/cpu'] = float(psutil.virtual_memory().used) / float(1073741824)
        metrics['memory/gpu'] = float(tc.memory_reserved(output.device)) / float(1073741824)
        metrics["memory/gpu_max"] = float(tc.max_memory_allocated(output.device)) / float(1073741824)

        return metrics


    def valid_batch_metrics(self, output, target):
        metrics = {}
        d = target-output
        d_mean = torch.mean(d)[None] 
        d_std  = torch.std(d)
        #d = self.EnergyRegression(output, target)
        if self.valid_mean is None:
            self.valid_mean = d_mean    
        else:
            self.valid_mean = d #torch.cat((self.valid_mean,d_mean), dim=0)


    def epoch_metrics(self):
        '''Function to calculate metrics for each epoch'''
        metrics = {}
        metrics['mean/epoch'] = { 
                'train' : torch.mean(self.train_mean), 
                'valid' : torch.mean(self.valid_mean)
        }
        metrics['std/epoch'] = { 
                'train' : torch.std(self.train_mean), 
                'valid' : torch.std(self.valid_mean)
        }
        del self.train_mean
        del self.valid_mean
        #metrics['time/epoch'] = {
        #    'train': self.train_end - self.epoch_start,
        #    'valid': 0 #self.valid_end - self.train_end
        #}
        return metrics 



