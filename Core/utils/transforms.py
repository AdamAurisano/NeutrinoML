import torch

def dense_to_sparse(dense):
    '''Transform dense NumPy array into MinkowskiEngine SparseTensor'''

    import MinkowskiEngine as ME
    sparse = torch.FloatTensor(dense).to_sparse()
    ret = ME.SparseTensor(sparse._values().unsqueeze(dim=-1), sparse._indices().T.int())
    
    return ret
