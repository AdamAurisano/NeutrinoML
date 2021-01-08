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

def collate_sparse_minkowski_2stack(batch):
  import MinkowskiEngine as ME
  x_coords = ME.utils.batched_coordinates([d['xcoords'] for d in batch])
  x_feats  = torch.cat([d['xfeats'] for d in batch])
  y_coords = ME.utils.batched_coordinates([d['ycoords'] for d in batch])
  y_feats  = torch.cat([d['yfeats'] for d in batch])
  y        = torch.stack([d['truth'] for d in batch])
  ret = { 'sparse': [x_feats, x_coords, y_feats, y_coords], 'y': y }
  return ret

def collate_dense_2stack(batch):
  xview = torch.stack([ x[0] for x in batch ])
  yview = torch.stack([ x[1] for x in batch ])
  y = torch.stack([ x[2] for x in batch ])
  return { 'xview': xview, 'yview': yview, 'y': y }
