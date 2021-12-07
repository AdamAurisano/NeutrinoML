# Simple PyTorch geometric dataset for spacepoint graphs
import glob, torch, torch_geometric as tg
from time import time
from .simple import SimpleDataset

class SPData(tg.data.Data):

  def __inc__(self, key, value, *args, **kwargs):
    if key == 'edge_index_u':
      return self.x_u.size(0)
    if key == 'edge_index_v':
      return self.x_v.size(0)
    if key == 'edge_index_y':
      return self.x_y.size(0)
    if key == 'edge_index_3d_u':
      return torch.tensor([[self.x_u.size(0)], [self.n_sp]])
    if key == 'edge_index_3d_v':                                      
      return torch.tensor([[self.x_v.size(0)], [self.n_sp]])              
    if key == 'edge_index_3d_y':                                            
      return torch.tensor([[self.x_y.size(0)], [self.n_sp]])                    
    else:
      return super().__inc__(key, value, *args, **kwargs)

class SpacePointDataset(SimpleDataset):
    
  def __getitem__(self, idx):
    filename = self.trainfiles[int(idx)]
    data = torch.load(filename)

    # temp fix
    row, col = data.edge_index_3d_u
    row1, col1 = data.edge_index_3d_v
    row2, col2 = data.edge_index_3d_y
    
    m1 = torch.max(col).detach() if len(col) > 0 else 0
    m2 = torch.max(col1).detach() if len(col1) > 0 else 0
    m3 = torch.max(col2).detach() if len(col2) > 0 else 0
    
    data.n_sp = max(m1,m2,m3) + 1

    data.num_nodes = data.y_s_u.shape[0] + data.y_s_v.shape[0] + data.y_s_y.shape[0]

    for p in [ "u", "v", "y" ]:
      data[f"x_{p}"] = data[f"x_{p}"][:,[1,2,7,8]]

    # terrible hell hack fix
    if data.edge_index_u.shape[1] == 2 and \
       data.edge_index_v.shape[1] == 2 and \
       data.edge_index_y.shape[1] == 2:
      data.edge_index_u = data.edge_index_u.transpose(0,1)
      data.edge_index_v = data.edge_index_v.transpose(0,1)
      data.edge_index_y = data.edge_index_y.transpose(0,1)

    if self.label_map is not None:
      data.y_s_u = self.label_map[data.y_s_u].long()
      data.y_s_v = self.label_map[data.y_s_v].long()
      data.y_s_y = self.label_map[data.y_s_y].long()

    if type(data) == tg.data.Data: 
      sp = SPData()
      sp.__dict__.update(data.__dict__)
      return sp
    else: 
      return SPData(**data)

  def feature_norm(self):
    # load first input
    first = self[0]
    print(first)

