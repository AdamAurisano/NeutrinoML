import torch
import MinkowskiEngine as ME
from SparseProtoDUNE.scripts.functions import * 

def offset_metric(pred_offset, true_offset, voxid):
  mask = voxid !=0
  offset_norm = torch.linalg.norm((pred_offset[mask] - true_offset[mask]),dim=1)
  return offset_norm.detach().cpu().numpy()

def instance_segementation_metrics_batch(batch_target, batch_output, th):
    #prediction
    pred_htm = batch_output['center_pred']
    pred_offsets = batch_output['offset_pred']
    
    #truth
    true_htm = batch_target['htm']
    true_offsets = batch_target['offset']
    true_med = batch_target['medoids']
    true_id = batch_target['voxId']
    n = batch_target['meta']
   
    ############## offset prediction metric ############
    offset_batch_metric = offset_metric(pred_offsets.F, true_offsets, true_id)  
       
    ############## center prediction metric ############
    ret = [0,0]
    kernel = 7
    #loop over individual maps 
    for k in range(pred_htm.C[:,0].max() +1):
        mask = (pred_htm.C[:,0] == k)
        p_htm = ME.SparseTensor(pred_htm.F[mask],pred_htm.C[mask]) ## sparse htm
        pred_med = get_medoid_prediction_sparse(p_htm,kernel,th,None)
        #get true medoids
        topk, _index =torch.topk(true_htm[mask], dim=0, k=n[k])
        t_med = pred_htm.C[mask,1:][_index.flatten().squeeze()]
        if n[k] ==1:
            t_med =  torch.reshape(t_med,(1,3))
        metrics, sorted_medoids = medoid_pred_metric(t_med, pred_med)
        ret += metrics
    return {'medoid_batch_metrics': ret/(k+1), 'offset_batch_metric' : offset_batch_metric}  
   

def panoptic_segmentation_metrics(true_id, reco_id):
    ''' script to calculate purity and completeness for panoptic segmentation results.
        true_id : true panoptic labels
        reco_id : reconstructed panoptic labels
        ----------
        Returns:
        completeness, purity
        '''

    event_completeness = 0
    event_purity = 0
    n_objects = torch.unique(true_id).shape[0]-1
    for k in torch.unique(true_id):
        #conmpute metrics only on "things"
        if k ==0: continue 
        true_mask = true_id == k
        reco_mask = reco_id == k
        correct = (true_id[true_mask] == reco_id[true_mask]).sum()
        if reco_mask.sum() ==0:
            purity = 0
            completeness = 0
        else:
            purity = correct/reco_mask.sum()
            completeness = correct/true_mask.sum()
    return np.array([event_completeness,  event_purity],dtype=np.float32)/n_objects 
