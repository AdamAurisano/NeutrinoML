def arrange_sparse(data, device):
    """Arrange ground truth for a batch of sparse pixel maps"""
    return data["y"].to(device)

def sparse_semantic_truth(data, device):
    """Arrange semantic segmentation ground truth for a batch of sparse pixel maps"""
    return data["y"].to(device)

def sparse_panoptic_truth(data, device):
    """Arrange panoptic segmentation ground truth for a batch of sparse pixel maps"""
    return { 'sem_seg': data["y"].to(device),
             'htm': data["htm"].to(device),
             'offset': data['offset'].to(device),
             'voxId': data['voxId'].to(device),
             'medoids': data['medoids'].to(device),
             'meta': data['meta']} 

def arrange_dense(data, device):
    """Arrange ground truth for a batch of sparse pixel maps"""
    return data["y"].to(device)

def arrange_graph(data, device):
    """Arrange ground truth for a batch of graphs when using DataParallel"""
    return data.y.to(device)

def arrange_graph_3d(data, device):
    """Arrange ground truth for a batch of graphs when using DataParallel"""
    import torch
    return torch.cat([data.y_s_u, data.y_s_v, data.y_s_y], dim=0).to(device)

