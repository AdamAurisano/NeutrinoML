import torch, MinkowskiEngine as ME

def collate_sparse(batch):
  for idx, d in enumerate(batch):
    d['c'] = torch.cat((d['c'], torch.LongTensor(d['c'].shape[0],1).fill_(idx)), dim=1)
  ret = { key: torch.cat([d[key] for d in batch], dim=0) for key in batch[0].keys() }
  return ret

def collate_sparse_minkowski(batch):
  coords = ME.utils.batched_coordinates([d['c'].int() for d in batch])
  feats = torch.cat([d['x'] for d in batch])
  y = torch.cat([d['y'] for d in batch])
  ret = { 'sparse': ME.SparseTensor(feats, coords), 'y': y }
  return ret

def collate_sparse_minkowski_2stack(batch):
  x_coords = ME.utils.batched_coordinates([d['xcoords'] for d in batch])
  x_feats  = torch.cat([d['xfeats'] for d in batch])
  y_coords = ME.utils.batched_coordinates([d['ycoords'] for d in batch])
  y_feats  = torch.cat([d['yfeats'] for d in batch])
  y        = torch.stack([d['truth'] for d in batch])
  ret = { 'sparse': [x_feats, x_coords, y_feats, y_coords], 'y': y }
  return ret