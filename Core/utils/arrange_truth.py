import torch

def arrange_sparse(data):
    """Arrange ground truth for a batch of sparse pixel maps"""
    return data["y"]

def sparse_semantic_truth(data):
    """Arrange semantic segmentation ground truth for a batch of sparse pixel maps"""
    return data["y"]

def sparse_panoptic_truth(data):
    """Arrange instance segmentation ground truth for a batch of sparse pixel maps"""
    ret ={'sem_seg':data["y"], 'ctr_htm':data["chtm"], 'offset':data['offset']} 
    return ret 

def arrange_dense(data):
    """Arrange ground truth for a batch of sparse pixel maps"""
    return data["y"]

def arrange_graph(data):
    """Arrange ground truth for a batch of graphs when using DataParallel"""
    return data.y
