import torch 
import numpy as np
from torch.nn import functional as f
import MinkowskiEngine as ME
from collections import Counter
#__all__ = ['get_center_prediction', 'get_panoptic_segmentation']

def get_center_prediction(htm,c, nms_kernel, threshold, top_k=None):
  '''Find medoid location from the heatmap''' 
  htm = f.threshold(htm, threshold, -1)
  htm = torch.reshape(htm,(htm.shape[1],htm.shape[0]))
  nms_padding = (nms_kernel - 1) // 2
  htm = htm.unsqueeze(0)
  pooled_htm = f.max_pool1d(htm, kernel_size=nms_kernel, stride=1, padding=nms_padding)
  #pooled_htm = F.avg_pool1d(htm, kernel_size=nms_kernel, stride=1, padding=nms_padding)
  htm[htm != pooled_htm] =-1
  # finding center candidates
  ctr_all = torch.nonzero(htm > 0)
  if top_k is None:
    return c[ctr_all[:,2]]
  elif ctr_all.shape[0] < top_k:
    return c[ctr_all[:,2]]
  else:
  # find top k centers.
    top_k_scores, _ = torch.topk(torch.flatten(htm), top_k)
    #return top_k_scores, _
    return c[torch.nonzero(htm >= top_k_scores[-1])[:,2]]


       
def get_medoid_prediction_sparse(htm, nms_kernel, threshold, top_k=None):
    '''Find the medoid location from the center heatmap
       ------- 
       Returns:
       torch.Tensor 
       '''
    c = htm.C[:,1:]
    ME_th = ME.MinkowskiThreshold(threshold,-1) 
    max_pool = ME.MinkowskiMaxPooling(
            kernel_size=nms_kernel,
            stride=1,
            dilation=1,
            kernel_generator=None,
            dimension=3,
            )
    htm = ME_th(htm)
    pooled_htm =  max_pool(htm)
    htm.F[htm.F != pooled_htm.F] = -1
    # finding center candidates
    ctr_all = torch.nonzero(htm.F > 0)
    if top_k is None:
        return c[ctr_all[:,0]]
    elif ctr_all.shape[0] < top_k:
        return c[ctr_all[:,0]]
    else:
    # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(htm.F), top_k)
        return c[torch.nonzero(htm.F >= top_k_scores[-1])[:,0]]

def group_pixels(c,offset,ctr):
  voxel_id = []
  for ci,Oi in zip(c,offset):
     _list = []
     for _ctr in ctr:
       if _ctr.sum() == -2997:
           continue
       Diff =   ci-_ctr -Oi  #_ctr -(ci+Oi)
       #_list.append(np.linalg.norm(Diff))
       _list.append(torch.norm(Diff))
     Id = np.argmin(_list)
     voxel_id.append(Id+1)
  voxel_id = torch.IntTensor(voxel_id)
  return voxel_id

def get_instance_segmentation(c,offset,ctr, sem_seg):
  voxel_id = group_pixels(c,offset,ctr)
  thing_list = [0,3,5,6]
  thing_seg = torch.zeros_like(sem_seg)
  for thing_class in thing_list:
    thing_seg[sem_seg == thing_class] = 1
  
  return thing_seg*voxel_id

def merge_semantic_and_instance(sem_seg, voxel_id):
    class_name = ["shower","delta","diffuse","hip","michel","mu","pi"]
    thing_list = [0,3,5,6]
    label_divisor = 100
    pan_seg = torch.zeros_like(sem_seg)
    semantic_thing_seg = torch.zeros_like(sem_seg)
    ## keep track of instance id for each class 
    class_id_tracker = Counter()
    particle_tracker = Counter()
    for thing_class in thing_list:
        semantic_thing_seg[sem_seg ==thing_class] =1
        ins_seg = voxel_id*semantic_thing_seg
    instance_ids = torch.unique(ins_seg)
    for ins_id in instance_ids:
        if ins_id == 0: continue # avoid background
        thing_mask = (ins_seg == ins_id) & (semantic_thing_seg == 1)
        class_id, _ = torch.mode(sem_seg[thing_mask]) # most commun value
        class_id_tracker[class_name[class_id.item()]] += 1
        new_ins_id = class_id_tracker[class_name[class_id.item()]]
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id
        particle_tracker[class_name[class_id]+str(new_ins_id)] =thing_mask
        
    return pan_seg, class_id_tracker, particle_tracker 

def get_semantic_segmentation(sem_seg):
  sem_label = sem_seg.argmax(dim=1) 

  return sem_label

def get_panoptic_segmentation(htm, c, nms_kernel=3, top_k=10):
  print(nms_kernel,htm.shape,top_k,c.shape)
  ctr = get_center_prediction(htm,c,nms_kernel,top_k)
  print(ctr)
 # voxel_id = group_pixels(c,offset,ctr)
 # sem_label = get_semantic_segmentation(sem_seg)
 # pan_seg = merge_semantic_and_instance(sem_label,voxel_id)
  return ctr #ctr, pan_seg

def medoid_pred_metric(true_med, pred_med):
    '''  script to calculate purity and efficiency for the medoid prediction head.
         Also sort the predicted medoids with respect to the true medoids
        ------- 
        Returns:
        tuple (putiry, efficiency)'''

    sorted_pred_medoids = torch.ones_like(true_med)*(-999)
    empty = 0 # N of instances with no pred medoid
    plus1 = 0 # N of instances with more than 1 medoid
    one_med = 0 # N of instances with  1 medoid
    #initial values
    purity = 0 
    efficiency = 0
    if pred_med.shape[0] ==0: 
        return np.array([purity, efficiency],dtype=np.float32)
    for k in pred_med:
        a = torch.linalg.norm((true_med-k).float(),dim=1)
        correct_med = a <7.6811
        index = None
        if correct_med.sum().item() == 0:
            empty +=1
        elif correct_med.sum().item() == 1:
            one_med +=1
            index = a.argmin()
        elif correct_med.sum().item() >1:
            one_med +=1
            index = a.argmin()
        sorted_pred_medoids[index] = k
    purity = one_med/pred_med.shape[0]
    efficiency = one_med/true_med.shape[0]
    #sorted_pred_medoids = 
    return np.array([purity, efficiency], dtype=np.float32), sorted_pred_medoids


