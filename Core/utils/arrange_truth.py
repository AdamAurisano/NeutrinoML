import torch

def arrange_sparse(data):
    '''Arrange ground truth for a batch of sparse pixel maps'''
    return data['y']

def arrange_graph(data):
    '''Arrange ground truth for a batch of graphs when using DataParallel'''
    return data.y

# def arrange_graph_parallel(data):
#     return torch.cat([ d.y for d in data ])