import torch

def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    y_true = y_true[:,:9]
    # Normalise the loss!
    loss = -(y_true * torch.log(y_pred))
    weights = torch.zeros(y_true.shape[1]).to(y_true.device)
    class_sum = y_true.sum(dim=0)
    mask = (class_sum>0)
    weights[mask] = y_true[:,mask].shape[0]/(y_true.shape[1]*class_sum[mask])
    weighted_loss = weights[None,:] * loss
    return weighted_loss.sum(dim=1).mean()
    
def cross_entropy(y_pred, y_true, weighted=True):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    loss = -((y_true * torch.log(y_pred)) + ((1-y_true) * torch.log(1-y_pred)))
    ### weigts 
    weights = torch.ones_like(y_true).to(y_true.device)
    f_true = y_true.flatten()
    N = f_true.shape[0] # total number of voxels
    n = (f_true>0.6).sum().item() # meaningful voxels
    weights[f_true>0.6] = N/(2*n)
    if weighted == True:
        return (loss*weights).mean()
    else:
        return loss.mean()

def categorical_focal_loss(y_pred, y_true, gamma=2):
    '''Focal loss function for multiclass classification with integer labels. '''
    #weigths 
    weights = torch.ones(y_true.shape[1]).to(y_true.device)
    class_sum = y_true.sum(dim=0)
    mask = (class_sum>0)
    #weights[mask] = torch.sqrt(y_true[:,mask].shape[0]/(y_true.shape[1]*class_sum[mask]))
    weights[mask] = y_true[:,mask].shape[0]/(y_true.shape[1]*class_sum[mask])
    w = torch.gather(weights, 0, y_true.argmax(dim=1))
    ## loss calculation 
    y_true = y_true.argmax(dim=1)
    pt = torch.gather(y_pred,1,y_true[:,None]) #model's estimated probability for the true label 
    pt = torch.clamp(pt, 1e-9, 1 - 1e-9)
    loss = -((1-pt)**gamma) * torch.log(pt)
    loss *= w[:,None]   #weighted loss 
    return loss.mean()


def focal_loss(y_pred, y_true, gamma=2):
    '''Focal loss function for multiclass classification with integer labels. '''
    ## loss calculation 
    pt = torch.gather(y_pred,1,y_true[:,None]) #model's estimated probability for the true label 
    pt = torch.clamp(pt, 1e-9, 1 - 1e-9)
    loss = -((1-pt)**gamma) * torch.log(pt)
    return loss.mean()


def generalized_dice(y_pred, y_true):
    weights = torch.zeros(y_true.shape[1]).to(y_true.device)
    class_sum = y_true.sum(dim=0)
    mask = (class_sum>0)
    weights[mask] = 1/y_true[:,mask].sum(dim=0)**2
    num = y_pred*y_true
    num = (weights*num)[:,mask].sum(dim=0)
    den = (y_pred + y_true)
    den = (weights*den)[:,mask].sum(dim=0)
    dice_coef = 2*num.sum()/den.sum()
    val = 1 -dice_coef
    return val
