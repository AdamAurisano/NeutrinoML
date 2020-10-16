import torch

def arrange_sparse(data):
    '''Arrange ground truth for a batch of sparse pixel maps'''
    return data['y']

def arrange_dense(data):
    '''Arrange ground truth for a batch of sparse pixel maps'''
    return data['y']

def arrange_graph(data):
    '''Arrange ground truth for a batch of graphs when using DataParallel'''
    return data.y
