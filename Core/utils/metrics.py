'''
Functions for calculating training metrics
'''

def metrics_semantic_segmentation(y_true, y_pred, class_names=None):
    '''Function to calculate training metrics for semantic segmentation'''
    metrics = { 'batch': {}, 'epoch': {} }
    w_pred = batch_output.argmax(dim=1)
    w_true = batch_target
    correct = (w_pred==w_true)
    metrics['batch']['Acc/batch'] = 100*correct.sum().float().item()/w_pred.shape[0]
    # Calculate accuracy individually for each class, if requested
    if class_names is not None:
        if len(class_names) != batch_output.shape[1]:
          raise Exception(f'Number of class names ({len(class_names)}) does not match shape of network output ({batch_output.shape[1]})!')
        #metrics['IndivAcc/acc_indiv = [ 100*((w_pred[correct]==i).sum().float()/(w_true==i).sum().float()).item() for i in range(batch_target.shape[1]) ]

