import torch

def collate_sparse(batch):
  for idx, d in enumerate(batch):
    d['c'] = torch.cat((d['c'], torch.LongTensor(d['c'].shape[0],1).fill_(idx)), dim=1)
  ret = { key: torch.cat([d[key] for d in batch], dim=0) for key in batch[0].keys() }
  return ret

def collate_sparse_minkowski(batch):
  import MinkowskiEngine as ME
  coords = ME.utils.batched_coordinates([d['c'].int() for d in batch])
  feats = torch.cat([d['x'] for d in batch])
  y = torch.cat([d['y'] for d in batch])
  ret = { 'f': feats, 'c': coords, 'y': y }
  return ret

def collate_sparse_minkowski_panoptic(batch):
  import MinkowskiEngine as ME
  coords = ME.utils.batched_coordinates([d['c'].int() for d in batch])
  feats = torch.cat([d['x'] for d in batch])
  y = torch.cat([d['y'] for d in batch])
  htm = torch.cat([d['htm'] for d in batch])
  offset =torch.cat([d['offset'] for d in batch])
  medoids =torch.cat([d['medoids'] for d in batch])
  voxId =torch.cat([d['voxId'] for d in batch])
  meta = [d['meta'] for d in batch]
  ret = { 'f': feats, 'c': coords, 'y': y, 'htm':htm, 'offset':offset, 'medoids':medoids, 'voxId':voxId, 'meta': meta}
  return ret 

def collate_sparse_minkowski_2stack(batch):
  import MinkowskiEngine as ME
  x_coords   = ME.utils.batched_coordinates([d['xcoords'] for d in batch])
  x_feats    = torch.cat([d['xfeats'] for d in batch])
  x_segtruth = torch.cat([d['xsegtruth'] for d in batch])
  x_instruth = torch.cat([d['xinstruth'] for d in batch])
  y_coords   = ME.utils.batched_coordinates([d['ycoords'] for d in batch])
  y_feats    = torch.cat([d['yfeats'] for d in batch])
  y_segtruth = torch.cat([d['ysegtruth'] for d in batch])
  y_instruth = torch.cat([d['yinstruth'] for d in batch])
  y          = torch.stack([d['evttruth'] for d in batch])
  ret = { 'sparse': [x_feats, x_coords, y_feats, y_coords], 'y': y }
  return ret

def collate_dense_2stack(batch):
  xview = torch.stack([ x[0] for x in batch ])
  yview = torch.stack([ x[1] for x in batch ])
  y = torch.stack([ x[2] for x in batch ])
  return { 'xview': xview, 'yview': yview, 'y': y }
