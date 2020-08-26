"""
This module defines a generic trainer for simple models and datasets. 
"""

# System
import time
import math

# Externals
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR
import tqdm, numpy as np, psutil

# Locals
from Core.models import get_model
from .base import base
from Core.loss import get_loss
from Core.activation import get_activation
from Core.optim import get_optim
from Core.metrics import get_metrics
from Core.utils import *
class SparseTrainer(base):
  '''Trainer code for basic classification problems with categorical cross entropy.'''

  def __init__(self, train_name='test1', summary_dir='summary', **kwargs):
    super(SparseTrainer, self).__init__(**kwargs)
    self.writer = SummaryWriter(f'{summary_dir}/{train_name}')

  def build_model(self, activation_params, optimizer_params, name='NodeConv',
      loss_func='cross_entropy', arrange_data = 'arrange_sparse_minkowski',
      metrics = 'SemanticSegmentation', metric_params=[],
      step_size=1, gamma=0.5, **model_args):
    '''Instantiate our model'''

    # Construct the model
    torch.cuda.set_device(self.device)
    model_args['activation'] = get_activation(**activation_params)
    self.model = get_model(name=name, **model_args)
    self.model = self.model.to(self.device)
    
    # Construct the loss function
    self.loss_func = get_loss(loss_func)

    # Construct the optimizer
    self.optimizer = get_optim(model_params=self.model.parameters(), **optimizer_params)
    self.lr_scheduler = StepLR(self.optimizer, step_size, gamma)

    self.metrics = get_metrics(metrics)(**metric_params[metrics])

    # Select function to arrange data
    self.arrange_data = arrange_data
  
  def load_state_dict(self, state_dict, **kwargs):
    '''Load state dict from trained model'''
    self.model.load_state_dict(torch.load(state_dict, map_location=f'cuda:{self.device}')['model'])

  def train_epoch(self, data_loader, **kwargs):
    '''Train for one epoch'''
    self.model.train()
    self.metrics.new_epoch()
    summary = dict()
    sum_loss = 0.
    start_time = time.time()
    # Loop over training batches
    batch_size = data_loader.batch_size
    n_batches = int(math.ceil(len(data_loader.dataset)/batch_size)) #if max_iters_train is None else max_iters_train
    t = tqdm.tqdm(enumerate(data_loader),total=n_batches)
    for i, data in t:
      self.optimizer.zero_grad()
      # Different input shapes for SparseConvNet vs MinkowskiEngine
      batch_input = self.arrange_data(data, self.device)
      batch_output = self.model(batch_input)
      batch_target = data['y'].to(batch_output.device)
      batch_loss = self.loss_func(batch_output, batch_target)
      batch_loss.backward()

      # Calculate accuracy
      metrics = self.metrics.train_batch_metrics(batch_output, batch_target) 
      
      self.optimizer.step()
      
      sum_loss += batch_loss.item()
      t.set_description("loss = %.5f" % batch_loss.item() )
      t.refresh() # to show immediately the update

      # add to tensorboard summary
      metrics = self.metrics.train_batch_metrics(batch_output, batch_target)
      if self.iteration%10 == 0:
        self.writer.add_scalar('Loss/batch', batch_loss.item(), self.iteration)
        for key, val in metrics.items(): self.writer.add_scalar(key, val, self.iteration)
      self.iteration += 1

    summary['lr'] = self.optimizer.param_groups[0]['lr']
    summary['train_time'] = time.time() - start_time
    summary['train_loss'] = sum_loss / n_batches
    self.logger.debug(' Processed %i batches', n_batches)
    self.logger.info('  Training loss: %.3f', summary['train_loss'])
    self.logger.info('  Learning rate: %.5f', summary['lr'])
    return summary

  @torch.no_grad()
  def evaluate(self, data_loader, max_iters_eval=None, **kwargs):
    '''Evaluate the model'''
    self.model.eval()
    summary = dict()
    sum_loss = 0
    start_time = time.time()
    # Loop over batches
    batch_size = data_loader.batch_size
    n_batches = int(math.ceil(len(data_loader.dataset)/batch_size))
    t = tqdm.tqdm(enumerate(data_loader),total=n_batches)
    for i, data in t:
      # Different input shapes for SparseConvNet vs MinkowskiEngine
      batch_input = self.arrange_data(data, self.device)
      batch_output = self.model(batch_input)
      batch_target = data['y'].to(batch_output.device)
      batch_loss = self.loss_func(batch_output, batch_target)
      sum_loss += batch_loss.item()
      self.metrics.valid_batch_metrics(batch_output, batch_target)
    summary['valid_time'] = time.time() - start_time
    summary['valid_loss'] = sum_loss / n_batches
    self.logger.debug(' Processed %i samples in %i batches',
                      len(data_loader.sampler), n_batches)
    self.logger.info('  Validation loss: %.3f' % (summary['valid_loss']))
    return summary

  def train(self, train_data_loader, n_epochs, valid_data_loader=None, **kwargs):
    '''Run the model training'''

    # Loop over epochs
    best_valid_loss = 99999
    self.iteration = 0
    for i in range(n_epochs):
      self.logger.info('Epoch %i' % i)
      summary = dict(epoch=i)
      # Train on this epoch
      sum_train = self.train_epoch(train_data_loader, **kwargs)
      summary.update(sum_train)
      # Evaluate on this epoch
      sum_valid = None
      if valid_data_loader is not None:
        sum_valid = self.evaluate(valid_data_loader, **kwargs)
        summary.update(sum_valid)

        if sum_valid['valid_loss'] < best_valid_loss:
          best_valid_loss = sum_valid['valid_loss']
          self.logger.debug('Checkpointing new best model with loss: %.3f', best_valid_loss)
          self.write_checkpoint(checkpoint_id=i,best=True)

      if self.lr_scheduler is not None: self.lr_scheduler.step()

      # Save summary, checkpoint
      self.save_summary(summary)
      if self.output_dir is not None:
        self.write_checkpoint(checkpoint_id=i)

      self.writer.add_scalars('loss/epoch', {
          'train': summary['train_loss'],
          'valid': summary['valid_loss'] }, i+1)
      metrics = self.metrics.epoch_metrics()
      for key, val in metrics.items(): self.writer.add_scalars(key, val, i+1)

    return self.summaries

def _test():
  t = SparseTrainer(output_dir='./')
  t.build_model()

