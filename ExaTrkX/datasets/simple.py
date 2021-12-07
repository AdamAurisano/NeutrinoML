# Simple PyTorch geometric dataset for spacepoint graphs
import os.path as osp, glob, torch, torch_geometric as tg

class SimpleDataset(tg.data.Dataset):

  def __init__(self, path, label_scheme, limit=None, **kwargs):
    self.path = path
    self.trainfiles = sorted(glob.glob(f'{self.path}/*.pt'))
    if limit is not None and len(self.trainfiles) > limit:
      self.trainfiles = self.trainfiles[0:limit]
    self.weight_file = osp.join(osp.split(path)[0], "weights", label_scheme+".pt")
    self.norm_file = osp.join(osp.split(path)[0], "weights", "norm.pt")
    if "label_map" in kwargs:
      self.label_map = torch.tensor(kwargs["label_map"]).int()
      
  def __len__(self):
    return len(self.trainfiles)
  
  def __getitem__(self, idx):
    filename = self.trainfiles[int(idx)]
    data = torch.load(filename)
    if type(data) != tg.data.Data: data = tg.data.Data(**data)
    if self.label_map is not None:
      data.y_s_u = self.label_map[y_s_u]
      data.y_s_v = self.label_map[y_s_v]
      data.y_s_y = self.label_map[y_s_y]

    data.num_nodes = data.y_s_u.shape[0] + data.y_s_v.shape[0] + data.y_s_y.shape[0]
    return data

  def __make_group_writeable(self, f):
    import os, stat
    os.chmod(f, os.stat(f).st_mode | stat.S_IWGRP)

  def load_weights(self, device):
    if osp.exists(self.weight_file):
      return torch.load(self.weight_file, map_location=device)
    else:
      return self.gen_weights(device)

  def gen_weights(self, device):
    import tqdm
    from torch_geometric.loader import DataLoader
    print("generating class weights from dataset and saving to file...")
    n_classes = self.label_map.max()+1
    train_loader = DataLoader(self, batch_size=200, num_workers=1)
    weights = torch.zeros(n_classes, device=device)
    for i, data in tqdm.tqdm(enumerate(train_loader), total=100):
      if i == 100: break
      data = data.to(device)
      for p in [ "u", "v", "y" ]:
        y = data[f"y_s_{p}"].unsqueeze(1).long()
        weights += torch.zeros([y.shape[0], n_classes], device=device).scatter_(1, y, 1).sum(dim=0)
    weights = float(weights.sum()) / (float(n_classes) * weights)
    torch.save(weights, self.weight_file)
    self.__make_group_writeable(self.weight_file)
    return weights

  def load_norm(self, device):
    if osp.exists(self.norm_file):
      return torch.load(self.norm_file, map_location=device)
    return self.gen_norm(device)

  def gen_norm(self, device):
    import tqdm
    from torch_geometric.loader import DataLoader
    print("generating feature norm from dataset and saving to file...")
    n_feats = self[0].x_u.shape[1]
    train_loader = DataLoader(self, batch_size=200, num_workers=1)
    norm = torch.empty([0, n_feats], device=device)
    for i, data in tqdm.tqdm(enumerate(train_loader)):
      data = data.to(device)
      for p in [ "u", "v", "y" ]:
        norm = torch.cat([norm, data[f"x_{p}"]], dim=0)
    mean = norm.mean(dim=0)
    std = norm.std(dim=0)
    torch.save([mean,std], self.norm_file)
    self.__make_group_writeable(self.norm_file)
    return mean, std

