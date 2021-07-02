def arrange_sparse(data, device):
    """Arrange ground truth for a batch of sparse pixel maps"""
    return data["y"].to(device)

def sparse_semantic_truth(data, device):
    """Arrange semantic segmentation ground truth for a batch of sparse pixel maps"""
    return data["y"].to(device)

def sparse_panoptic_truth(data, device):
    """Arrange panoptic segmentation ground truth for a batch of sparse pixel maps"""
    return { 'sem_seg': data["y"].to(device),
             'ctr_htm': data["chtm"].to(device),
             'offset': data['offset'].to(device) } 

def arrange_dense(data, device):
    """Arrange ground truth for a batch of sparse pixel maps"""
    return data["y"].to(device)

def arrange_graph(data, device):
    """Arrange ground truth for a batch of graphs when using DataParallel"""
    return data.y.to(device)
