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

from GraphDUNE.models import get_model
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

    def __init__(self, summary_dir='summary', **kwargs):
        super(GNNParallelTrainer, self).__init__(**kwargs)
        self.writer = SummaryWriter(summary_dir)

    def build_model(self, name='NodeConv', gpus=[0],
                    loss_func='binary_cross_entropy', pos_weight=1,
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
        elif loss_func == 'CrossEntropyLoss':
          self.loss_func = nn.CrossEntropyLoss()
        elif loss_func == 'WeightedBCE':
          self.loss_func = nn.BCELoss(reduction='none')
          self.weight = pos_weight
        else:
          self.loss_func = getattr(nn, loss_func)(pos_weight=torch.tensor(pos_weight).to(self.device))

        # Construct the optimizer
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

            self.optimizer.zero_grad()
            batch_output = self.model(data)
            batch_target = torch.cat([ d.y for d in data ]).to(batch_output.device)
            batch_loss = self.loss_func(batch_output, batch_target)
            if hasattr(self, 'weight'):
              batch_loss[(batch_target==1)] *= self.weight
              batch_loss = batch_loss.mean()
            batch_loss.backward()
            batch_loss_item = batch_loss.item()
            self.optimizer.step()

            mask = (batch_target==1)
            n_true = batch_target.sum().item()
            batch_acc = 100*(batch_output.round()==batch_target).sum() / batch_target.shape[0]
            true_acc = 100*(batch_output[mask].round()==batch_target[mask]).sum() / n_true
            false_acc = 100*(batch_output[~mask].round() == batch_target[~mask]).sum() / (batch_target.shape[0]-n_true)

            sum_loss += batch_loss.item()
            t.set_description("loss = %.5f" % batch_loss.item() )
            t.refresh() # Immediately show the update

            # Add to tensorboard summary
#             self.writer.add_scalar('Loss/batch', batch_loss, self.iteration)
#             self.writer.add_scalar('Accuracy/batch_all', batch_acc, self.iteration)
#             self.writer.add_scalar('Accuracy/batch_true', true_acc, self.iteration)
#             self.writer.add_scalar('Accuracy/batch_false', false_acc, self.iteration)
            self.iteration += 1

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
        sum_total = 0
        sum_true = 0
        sum_correct = 0
        sum_true_correct = 0
        sum_false_correct = 0
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
            if hasattr(self, 'weight'):
              batch_loss[(batch_target==1)] *= self.weight
              batch_loss = batch_loss.mean()
            sum_loss += batch_loss.item()
            # Count number of correct predictions
            mask = (batch_target==1)
            sum_total += batch_target.shape[0]
            sum_true += batch_target.sum().item()
            sum_correct += (batch_output.round()==batch_target).sum().item()
            sum_true_correct += (batch_output[mask].round()==batch_target[mask]).sum().item()
            sum_false_correct += (batch_output[~mask].round()==batch_target[~mask]).sum().item()           
        summary['valid_time'] = time.time() - start_time
        summary['valid_loss'] = sum_loss / (i + 1)
        summary['valid_acc'] = 100*sum_correct/sum_total
        summary['valid_acc_true'] = 100*sum_true_correct/sum_true
        summary['valid_acc_false'] = 100*sum_false_correct/(sum_total-sum_true)
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
                self.writer.add_scalar('Accuracy/valid_all', summary['valid_acc'], i+1)
                self.writer.add_scalar('Accuracy/valid_true', summary['valid_acc_true'], i+1)
                self.writer.add_scalar('Accuracy/valid_false', summary['valid_acc_false'], i+1)

        return self.summaries


def _test():
    t = GNNParallelTrainer(output_dir='./')
    t.build_model()

