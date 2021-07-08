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
import tqdm, numpy as np, psutil

# Locals
from Core.models import get_model
from .base import base
from Core.loss import get_loss
from Core.activation import get_activation
from Core.optim import get_optim
from Core.scheduler import get_scheduler
from Core.metrics import get_metrics
from Core.utils import *

class TrainerPanopticSeg(base):
  '''Trainer code for basic classification problems with categorical cross entropy.'''

  def __init__(self, train_name='test1', summary_dir='summary',
    empty_cache = None, **kwargs):
    super(TrainerPanopticSeg, self).__init__(train_name=train_name, **kwargs)
    self.writer = SummaryWriter(f'{summary_dir}/{train_name}')
    self.empty_cache = empty_cache

  def build_model(self, activation_params, optimizer_params, scheduler_params,
      loss_params, metric_params, name='NodeConv',
      arrange_data='arrange_sparse_minkowski', arrange_truth='arrange_sparse',
      **model_args):
    '''Instantiate our model'''

    # Construct the model
    torch.cuda.set_device(self.device)
    model_args['A'] = get_activation(**activation_params)
    self.model = get_model(name=name, **model_args)
    self.model = self.model.to(self.device)
    
    # Construct the loss functions
    self.semantic_loss = get_loss(**loss_params['SEMANTIC'])
    self.ctr_loss   = get_loss(**loss_params['CENTER'])
    self.offset_loss   = get_loss(**loss_params['OFFSET'])

    # Construct the optimizer
    self.optimizer = get_optim(model_params=self.model.parameters(), **optimizer_params)
    self.scheduler = get_scheduler(self.optimizer, **scheduler_params)

    # Configure metrics
    metrics=metric_params['metrics']
    metric_args = metric_params[metrics]
    self.metrics = get_metrics(metrics)(**metric_args)
    self.batch_metrics = metric_params['include_batch_metrics']

    # Select function to arrange data
    self.arrange_data = get_arrange_data(arrange_data)
    self.arrange_truth = get_arrange_truth(arrange_truth)
 
  def load_state_dict(self, state_dict, **kwargs):
    '''Load state dict from trained model'''
    self.model.load_state_dict(torch.load(state_dict, map_location=f'cuda:{self.device}')['model'])

  def train_epoch(self, data_loader, **kwargs): #lambda_weight, **kwargs):
    '''Train for one epoch'''
    self.model.train()
    self.metrics.new_epoch()
    summary = dict()
    sum_loss = 0.
    sum_sem_loss = 0.
    sum_ctr_loss = 0.
    sum_offset_loss = 0.
    start_time = time.time()
    # Loop over training batches
    batch_size = data_loader.batch_size
    n_batches = int(math.ceil(len(data_loader.dataset)/batch_size)) #if max_iters_train is None else max_iters_train
    t = tqdm.tqdm(enumerate(data_loader),total=n_batches)
    #detector_extend = torch.FloatTensor([350,600,700]).to(self.device)
    #looping over batches
    for i, data in t:
      ## data files
      #print(data_loader.dataset.dataset.data_files[i])
      self.optimizer.zero_grad()
      # input
      batch_input = self.arrange_data(data, self.device)
      #outputs 
      batch_output = self.model(batch_input)
      pred_htm = batch_output['center_pred'].F
      pred_semseg = batch_output['semantic_pred']
      pred_offset = batch_output['offset_pred'].F
     # pred_offset = 2*(pred_offset-0.5)
      # targets
      batch_target = self.arrange_truth(data)
      true_htm = batch_target['ctr_htm'].to(self.device)
      scale  = 1/true_htm.max().item()
      true_htm *= scale 
      true_semseg = batch_target['sem_seg'].to(self.device) 
      true_offset = (batch_target['offset']).to(self.device)
      #true_offset = (1.429e-3*batch_target['offset']).to(self.device)


     # print('pred', pred_offset)
      bk_mask = batch_target['voxId'] != 0 
      #LOSS FUNCTIONS
      _sem_loss = self.semantic_loss(pred_semseg, true_semseg)
      _ctr_loss = self.ctr_loss(pred_htm, true_htm)
      #_offset_loss = 150*self.offset_loss(pred_offset, true_offset)
      _offset_loss = 5e-2*self.offset_loss(pred_offset[bk_mask], true_offset[bk_mask])
       
      batch_loss = _sem_loss #_offset_loss +_sem_loss  #_ctr_loss + _offset_loss +  _sem_loss 
       
      
      #back propagation and optimization 
      batch_loss.backward()
      self.optimizer.step()

      if self.scheduler is not None:
        self.scheduler.step()
      
      sum_loss += batch_loss.item() 
      sum_sem_loss += _sem_loss.item() 
      sum_ctr_loss += _ctr_loss.item() 
      sum_offset_loss += _offset_loss.item() 
      t.set_description("loss = %.5f" % batch_loss.item())  #batch_loss.item() )
      t.refresh() # to show immediately the update

      # add to tensorboard summary
      if self.batch_metrics:
        #Calculate accuracy for semantic Segmentation
        metrics = self.metrics.train_batch_metrics(pred_semseg,true_semseg) 
        for key, val in metrics.items(): self.writer.add_scalar(key, val, self.iteration)
     
      if self.iteration%100 == 0:
        self.writer.add_scalar('loss/batch', batch_loss.item(), self.iteration)
        self.writer.add_scalar('sem_loss/batch', _sem_loss.item(), self.iteration)
        self.writer.add_scalar('center_loss/batch', _ctr_loss.item(), self.iteration)
        self.writer.add_scalar('offset_loss/batch', _offset_loss.item(), self.iteration)
      self.iteration += 1

      if self.empty_cache is not None and self.iteration % self.empty_cache == 0:
        torch.cuda.empty_cache()
       
    summary['lr'] = self.optimizer.param_groups[0]['lr']
    summary['train_time'] = time.time() - start_time
    summary['train_loss'] = sum_loss / n_batches
    summary['train_semantic_loss'] = sum_sem_loss / n_batches
    summary['train_center_loss'] = sum_ctr_loss / n_batches
    summary['train_offset_loss'] = sum_offset_loss / n_batches
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
    sum_sem_loss = 0
    sum_ctr_loss = 0
    sum_offset_loss = 0
    start_time = time.time()
    # Loop over batches
    batch_size = data_loader.batch_size
    n_batches = int(math.ceil(len(data_loader.dataset)/batch_size))
    t = tqdm.tqdm(enumerate(data_loader),total=n_batches)
    detector_extend = torch.FloatTensor([350,600,700]).to(self.device)
    for i, data in t:
      batch_input = self.arrange_data(data, self.device)

      #outputs 
      batch_output = self.model(batch_input)
      pred_htm = batch_output['center_pred'].F
      pred_semseg = batch_output['semantic_pred'] 
      pred_offset = batch_output['offset_pred'].F
     # pred_offset = 2*(pred_offset-0.5)
      
      # targets
      batch_target = self.arrange_truth(data)
      true_htm = batch_target['ctr_htm'].to(self.device)
      true_htm *= (0.5*(true_htm.max().item())) 
      true_semseg = batch_target['sem_seg'].to(self.device) 
      true_offset = (batch_target['offset']).to(self.device)
      #true_offset = (1.429e-3*batch_target['offset']).to(self.device)
      #loss calculation 
      _sem_loss = self.semantic_loss(pred_semseg, true_semseg)
      _ctr_loss = self.ctr_loss(pred_htm, true_htm)
      _offset_loss = 5e-2*self.offset_loss(pred_offset, true_offset)

      #if math.isnan(_distance_loss) or math.isnan(_center_loss):
      #    print('Nan distance loss', data_loader.dataset.dataset.data_files[i])
      #    bad_files(data_loader.dataset.dataset.data_files[i])
      #    continue 

      batch_loss = _sem_loss #_offset_loss + _sem_loss  #_ctr_loss + _offset_loss +  _sem_loss # Panoptic Segmentation Model 
     
      sum_loss += batch_loss.item()
      sum_sem_loss += _sem_loss.item()
      sum_ctr_loss += _ctr_loss.item()
      sum_offset_loss += _offset_loss.item()
      self.metrics.valid_batch_metrics(pred_semseg, true_semseg)
    
    summary['valid_time'] = time.time() - start_time
    summary['valid_loss'] = sum_loss / n_batches
    summary['valid_semantic_loss'] = sum_sem_loss / n_batches
    summary['valid_center_loss'] = sum_ctr_loss / n_batches
    summary['valid_offset_loss'] = sum_offset_loss / n_batches
    self.logger.debug(' Processed %i samples in %i batches',
                      len(data_loader.sampler), n_batches)
    self.logger.info('  Validation loss: %.3f' % (summary['valid_loss']))
    return summary

  def train(self, train_data_loader, n_epochs, valid_data_loader=None, sherpa_study=None, sherpa_trial=None, **kwargs):
    '''Run the model training'''

    # Loop over epochs
    best_valid_loss = 99999
    self.iteration = 0
    avg_loss = torch.zeros([n_epochs, 2]).to(self.device)
    lambda_weight = torch.ones([2, n_epochs]).to(self.device)

    for i in range(n_epochs):
      self.logger.info('Epoch %i' % i)
      self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], i+1)
      summary = dict(epoch=i)
      sum_train = self.train_epoch(train_data_loader, **kwargs) #lambda_weight[:,i], **kwargs)
     
     #multi-task weights
     # Train on this epoch and  apply Dynamic Weight Average
     # T =1 # Temperature, controls the softness of task weighting 
     # if i == 0 or i == 1:
     #   lambda_weight[:, i] = 1.0
     #   sum_train = self.train_epoch(train_data_loader,lambda_weight[:,i], **kwargs)
     #   avg_loss[i,0] = sum_train['train_center_loss'] 
     #   avg_loss[i,1] = sum_train['train_offset_loss']
     # else:
     #   avg_loss[i,0] = sum_train['train_center_loss'] 
     #   avg_loss[i,1] = sum_train['train_offset_loss']
     #   w_1 = avg_loss[i - 1, 0] / avg_loss[i - 2, 0]
     #   w_2 = avg_loss[i - 1, 1] / avg_loss[i - 2, 1]
     #   lambda_weight[0, i] = 2 * torch.exp(w_1 / T) / (torch.exp(w_1 / T) + torch.exp(w_2 / T)) 
     #   lambda_weight[1, i] = 2 * torch.exp(w_2 / T) / (torch.exp(w_1 / T) + torch.exp(w_2 / T)) 
     #   #print('epoch ', i)
     #   #print(lambda_weight[:,i])
     #   sum_train = self.train_epoch(train_data_loader, lambda_weight[:,i], **kwargs)


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

     # if self.scheduler is not None:
     #   self.scheduler.step(sum_valid["valid_loss"])

      # Save summary, checkpoint
      self.save_summary(summary)
      if self.output_dir is not None:
        self.write_checkpoint(checkpoint_id=i)

      self.writer.add_scalars('loss/epoch', {
          'train': summary['train_loss'],
          'valid': summary['valid_loss'] }, i+1)
      self.writer.add_scalars('sem_loss/epoch', {
          'train': summary['train_semantic_loss'],
          'valid': summary['valid_semantic_loss'] }, i+1)
      self.writer.add_scalars('center_loss/epoch', {
          'train': summary['train_center_loss'],
          'valid': summary['valid_center_loss'] }, i+1)
      self.writer.add_scalars('offset_loss/epoch', {
          'train': summary['train_offset_loss'],
          'valid': summary['valid_offset_loss'] }, i+1)
      metrics = self.metrics.epoch_metrics()
      if sherpa_study is not None and sherpa_trial is not None:
          sherpa_study.add_observation(
              trial=sherpa_trial,
              iteration=i,
              objective=metrics['acc/epoch']['valid'])
      for key, val in metrics.items(): self.writer.add_scalars(key, val, i+1)
    #_name = 'dwaT2' 
    #torch.save(lambda_weight,'/scratch/SparseProtoDUNE/dwa_files/'+_name+'.pt')

    return self.summaries

def _test():
  t = Trainer(output_dir='./')
  t.build_model()

