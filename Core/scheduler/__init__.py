'''
Module to load scheduler
'''

def get_scheduler(optim, scheduler, **params):
  from torch.optim import lr_scheduler
  ret = getattr(lr_scheduler, scheduler)
  return ret(optim, **params[scheduler])

