# Simple PyTorch geometric dataset for spacepoint graphs
import glob, torch, torch_geometric as tg
from time import time

class SimpleDataset(tg.data.Dataset):
    
    def __init__(self, path, **kwargs):
        self.path = path
        self.trainfiles = sorted(glob.glob(f'{self.path}/*.pt'))
        
    def __len__(self):
        return len(self.trainfiles)
    
    def __getitem__(self, idx):
        start = time()
        filename = self.trainfiles[int(idx)]
        data = torch.load(filename)
        data['x'] = torch.tensor(data['x']).float()
        if type(data.edge_index) is not torch.Tensor:
            data['edge_index'] = torch.tensor(data.edge_index)
        data['edge_index'] = data.edge_index.long()
        data['y'] = torch.tensor(data['y']).long()
        return data
