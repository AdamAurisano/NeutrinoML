"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import time
import math

# Externals
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
from torch.optim.lr_scheduler import LambdaLR, StepLR
import tqdm

from models import get_model
# Locals
from .base import base

def categorical_cross_entropy(y_pred, y_true):
    start = time.time()
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    # Normalise the loss!
    loss = torch.stack([-(y_true*torch.log(y_pred)),-((1-y_true)*torch.log(1-y_pred))], dim=-1)
    ##print(loss)
    #print(loss.shape)
    #print(y_true)
    weights = torch.FloatTensor([y_true.shape[0]/(2*y_true.sum()),y_true.shape[0]/(2*(1-y_true).sum())]).to(loss.device)
    #print(weights)
    #weights = torch.cat([y_true.shape[0]/(2*y_true.sum()),y_true.shape[0]/(2*(1-y_true.sum()))])
    #print (weights.shape)
    weighted_loss = weights[None,:] * loss
    ret = weighted_loss.sum(dim=1).mean()
    print(f'Loss function took {time.time()-start} seconds.')
    return ret

class GNNParallelTrainer(base):
    """Trainer code for basic classification problems with binomial cross entropy."""

    def __init__(self, real_weight=1, fake_weight=1, summary_dir='summary', **kwargs):
        super(GNNParallelTrainer, self).__init__(**kwargs)
        self.real_weight = real_weight
        self.fake_weight = fake_weight
        self.writer = SummaryWriter(summary_dir)

    def build_model(self, name='NodeConv', gpus=[0],
                    loss_func='binary_cross_entropy',
                    optimizer='Adam', learning_rate=0.01,
                    step_size=5, gamma=0.5, **model_args):
        """Instantiate our model"""

        # Construct the model
        self.device = f'cuda:{gpus[0]}'
        self.model = get_model(name=name, **model_args)
        self.model = torch_geometric.nn.DataParallel(self.model, device_ids=gpus).to(self.device)

        # Construct the loss function
        if loss_func == 'categorical_cross_entropy':
            self.loss_func = categorical_cross_entropy
        else:
            self.loss_func = getattr(nn.functional, loss_func)

        # Construct the optimizer
        print('Learning rate is', learning_rate)
        self.optimizer = getattr(torch.optim, optimizer)(
            self.model.parameters(), lr=learning_rate)

        self.lr_scheduler = StepLR(self.optimizer, step_size, gamma)


    # @profile
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        summary = dict()
        sum_loss = 0.
        start_time = time.time()
        # Loop over training batches
        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        t = tqdm.tqdm(enumerate(data_loader),total=int(math.ceil(total/batch_size)))
        for i, data in t:
        #for i, data in enumerate(data_loader):
            #data = data.to(self.device)
            self.optimizer.zero_grad()
            batch_output = self.model(data)
            batch_target = torch.cat([ d.y for d in data ]).to(batch_output.device)
            #batch_weights_real = batch_target*self.real_weight
            #batch_weights_fake = (1 - batch_target)*self.fake_weight
            #batch_weights = batch_weights_real + batch_weights_fake
            #frac = batch_target.sum(dim=0) / batch_target.shape[0]
            #batch_weights = 2*((frac*(1-batch_target))+((1-frac)*batch_target))
            batch_loss = self.loss_func(batch_output, batch_target)
            batch_loss.backward()
            batch_loss_item = batch_loss.item()
            self.optimizer.step()

            mask = (batch_target==1)
            n_true = batch_target.sum().item()
            true_acc = 100*(batch_output[mask].round()==batch_target[mask]).sum() / n_true
            false_acc = 100*(batch_output[~mask].round() == batch_target[~mask]).sum() / (batch_target.shape[0]-n_true)

            #print('true accuracy', true_acc, 'false_accuracy', false_acc)

            sum_loss += batch_loss.item()
            t.set_description("loss = %.5f" % batch_loss.item() )
            t.refresh() # to show immediately the update

            # add to tensorboard summary
            self.writer.add_scalar('Loss/batch', batch_loss, self.iteration)
            self.writer.add_scalar('Accuracy/true', true_acc, self.iteration)
            self.writer.add_scalar('Accuracy/false', false_acc, self.iteration)
            self.iteration += 1
            #self.logger.debug('  batch %i, loss %f', i, batch_loss.item())

        if self.lr_scheduler != None:
            self.lr_scheduler.step()

        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_time'] = time.time() - start_time
        summary['train_loss'] = sum_loss / (i + 1)
        self.logger.debug(' Processed %i batches', (i + 1))
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        # self.logger.info('  Learning rate: %.5f', summary['lr'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        sum_true = 0
        sum_false = 0
        sum_misid = 0
        sum_missed = 0
        sum_total_true = 0
        sum_total_false = 0
        start_time = time.time()
        # Loop over batches
        total = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        t = tqdm.tqdm(enumerate(data_loader),total=int(math.ceil(total/batch_size)))
        for i, data in t:
            # self.logger.debug(' batch %i', i)
            batch_output = self.model(data)
            batch_target = torch.cat([ d.y for d in data ]).to(batch_output.device)
            batch_loss = self.loss_func(batch_output, batch_target)
            sum_loss += batch_loss.item()
            # Count number of correct predictions
            matches = ((batch_output > 0.5) == (batch_target > 0.5))
            sum_true += ((batch_output > 0.5) & (batch_target > 0.5)).sum().item()
            sum_false += ((batch_output < 0.5) & (batch_target < 0.5)).sum().item()
            sum_misid += ((batch_output > 0.5) & (batch_target < 0.5)).sum().item()
            sum_missed += ((batch_output < 0.5) & (batch_target > 0.5)).sum().item()
            sum_total_true += (batch_target > 0.5).sum().item()
            sum_total_false += (batch_target < 0.5).sum().item()
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            #self.logger.debug(' batch %i loss %.3f correct %i total %i',
            #                  i, batch_loss.item(), matches.sum().item(),
            #                  matches.numel())                           
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = sum_correct / sum_total
        summary['valid_true'] = sum_true / sum_total
        summary['valid_false'] = sum_false / sum_total
        summary['valid_misid'] = sum_misid / sum_total
        summary['valid_missed'] = sum_missed / sum_total
        summary['valid_true_eff'] = sum_true / sum_total_true
        summary['valid_false_eff'] = sum_false / sum_total_false
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""

        # Loop over epochs
        best_valid_loss = 99999
        self.iteration = 0
        for i in range(n_epochs):
            self.logger.info('Epoch %i' % i)
            summary = dict(epoch=i)
            # Train on this epoch
            sum_train = self.train_epoch(train_data_loader)
            summary.update(sum_train)
            # Evaluate on this epoch
            sum_valid = None
            if valid_data_loader is not None:
                sum_valid = self.evaluate(valid_data_loader)
                summary.update(sum_valid)

                if sum_valid['valid_loss'] < best_valid_loss:
                    best_valid_loss = sum_valid['valid_loss']
                    self.logger.debug('Checkpointing new best model with loss: %.3f', best_valid_loss)
                    self.write_checkpoint(checkpoint_id=i,best=True)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Save summary, checkpoint
            self.save_summary(summary)
            if self.output_dir is not None:
                self.write_checkpoint(checkpoint_id=i)

            self.writer.add_scalar('Loss/train', summary['train_loss'], i+1)
            if valid_data_loader is not None:
                self.writer.add_scalar('Loss/valid', summary['valid_loss'], i+1)
                self.writer.add_scalar('Acc/valid', summary['valid_acc'], i+1)
                self.writer.add_scalar('Acc/valid_true_eff', summary['valid_true_eff'], i+1)
                self.writer.add_scalar('Acc/valid_false_eff', summary['valid_false_eff'], i+1)

        return self.summaries


def _test():
    t = GNNParallelTrainer(output_dir='./')
    t.build_model()

