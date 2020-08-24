import torch, MinkowskiEngine as ME

def arrange_sparse(data, device):
    batch_input = (data['c'].to(device), data['x'].to(device), batch_size)
    return batch_input

def arrange_sparse_minkowski(data, device):
    batch_input = data['sparse'].to(device)
    return batch_input

def arrange_sparse_minkowski_2stack(data, device):
    batch_input = [ data['sparse'][0].to(device),
                    data['sparse'][1],   
                    data['sparse'][2].to(device),
                    data['sparse'][3] ]
    return batch_input
