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
from TauRNN.models import get_model
from .base import base

class Trainer(base):
  '''Trainer code for basic classification problems with categorical cross entropy.'''

  def __init__(self, train_name='test1', summary_dir='summary', **kwargs):
    super(Trainer, self).__init__(**kwargs)
    self.writer = SummaryWriter(f'{summary_dir}/{train_name}')

  def build_model(self, name='TauRNN', loss_func='cross_entropy',
      optimizer='Adam', learning_rate=0.01, weight_decay=0.01,
      step_size=1, gamma=0.5, class_names=[], **model_args): #state_dict=None, **model_args):
    '''Instantiate our model'''

    # Construct the model
    torch.cuda.set_device(self.device)
    self.model = get_model(name=name, **model_args)
    self.model = self.model.to(self.device)

    # Construct the loss function
    if loss_func == 'categorical_cross_entropy':
      self.loss_func = categorical_cross_entropy
    elif loss_func == 'CrossEntropyLoss':
      self.loss_func = nn.CrossEntropyLoss()
    else:
      self.loss_func = getattr(nn.modules.loss, loss_func)()

    # Construct the optimizer
    self.optimizer = getattr(torch.optim, optimizer)(
      self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    self.lr_scheduler = StepLR(self.optimizer, step_size, gamma)

    self.class_names = class_names

  def load_state_dict(self, state_dict, **kwargs):
    '''Load state dict from trained model'''
    self.model.load_state_dict(torch.load(state_dict, map_location=f'cuda:{self.device}')['model'])

  def train_epoch(self, data_loader, **kwargs):
    '''Train for one epoch'''
    self.model.train()
    summary = dict()
    sum_loss = 0.
    start_time = time.time()
    # Loop over training batches
    batch_size = data_loader.batch_size
    n_batches = int(math.ceil(len(data_loader.dataset)/batch_size))
    t = tqdm.tqdm(enumerate(data_loader),total=n_batches)
    for i, data in t:
      self.optimizer.zero_grad()
      batch_input = ( data['x_fixed'].to(self.device),
                      data['x_point_var'].to(self.device),
                      data['x_hit_var'].to(self.device) )
      batch_output = self.model(batch_input)
      batch_target = data['y'].to(batch_output.device)
      batch_loss = self.loss_func(batch_output, batch_target)
      batch_loss.backward()

      # Calculate accuracy
      w_pred = batch_output.round()
      w_true = batch_target.argmax(dim=1) if batch_target.ndim == 2 else batch_target
      correct = (w_pred==w_true)
      batch_acc = 100*correct.sum().float().item()/w_pred.shape[0]
      #acc_indiv = [ 100*((w_pred[correct]==i).sum().float()/(w_true==i).sum().float()).item() for i in range(batch_target.shape[1]) ]

      self.optimizer.step()

      sum_loss += batch_loss.item()
      t.set_description("loss = %.5f" % batch_loss.item() )
      t.refresh() # to show immediately the update

      # add to tensorboard summary
      if self.iteration%100 == 0:
        self.writer.add_scalar('Loss/batch', batch_loss.item(), self.iteration)
        self.writer.add_scalar('Acc/batch', batch_acc, self.iteration)
      #for name, acc in zip(self.class_names, acc_indiv):
      #  self.writer.add_scalar(f'batch_acc/{name}', acc, self.iteration)
      #self.writer.add_scalar('Memory usage', psutil.virtual_memory().used, self.iteration)
      self.iteration += 1

    summary['lr'] = self.optimizer.param_groups[0]['lr']
    summary['train_time'] = time.time() - start_time
    summary['train_loss'] = sum_loss / n_batches
    self.logger.debug(' Processed %i batches', n_batches)
    self.logger.info('  Training loss: %.3f', summary['train_loss'])
    self.logger.info('  Learning rate: %.5f', summary['lr'])
    return summary

  @torch.no_grad()
  def evaluate(self, data_loader, **kwargs):
    '''Evaluate the model'''
    self.model.eval()
    summary = dict()
    sum_loss = 0
    sum_correct = 0
    sum_correct_indiv = np.zeros(len(self.class_names))
    sum_total = 0
    sum_total_indiv = np.zeros(len(self.class_names))
    start_time = time.time()
    # Loop over batches
    batch_size = data_loader.batch_size
    n_batches = int(math.ceil(len(data_loader.dataset)/batch_size))
    t = tqdm.tqdm(enumerate(data_loader),total=n_batches)
    for i, data in t:
      batch_input = ( data['x_fixed'].to(self.device),
                      data['x_point_var'].to(self.device),
                      data['x_hit_var'].to(self.device))
      batch_output = self.model(batch_input)
      batch_target = data['y'].float().to(batch_output.device)
      batch_loss = self.loss_func(batch_output, batch_target)
      print(batch_output)
      print(batch_target)
      sum_loss += batch_loss.item()
      w_pred = batch_output.round()
      w_true = batch_target.argmax(dim=1) if batch_target.ndim == 2 else batch_target
      correct = (w_pred==w_true)
      sum_correct += correct.sum().float().item()
      sum_total += w_pred.shape[0]
    summary['valid_time'] = time.time() - start_time
    summary['valid_loss'] = sum_loss / n_batches
    summary['valid_acc'] = 100 * sum_correct / sum_total
    if len(self.class_names) > 0:
      for name, corr, tot in zip(self.class_names, sum_correct_indiv, sum_total_indiv):
        summary[f'{name}_acc'] =100 * corr / tot
    self.logger.debug(' Processed %i samples in %i batches',
                      len(data_loader.sampler), n_batches)
    self.logger.info('  Validation loss: %.3f acc: %.3f' %
                     (summary['valid_loss'], summary['valid_acc']))
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

      self.writer.add_scalar('Loss/train', summary['train_loss'], i+1)
      if valid_data_loader is not None:
        self.writer.add_scalar('Loss/valid', summary['valid_loss'], i+1)
        self.writer.add_scalar('Acc/valid', summary['valid_acc'], i+1)
        for name in self.class_names:
          self.writer.add_scalar(f'valid_acc/{name}', summary[f'{name}_acc'], i+1)

    return self.summaries

def _test():
  t = Trainer(output_dir='./')
  t.build_model()

