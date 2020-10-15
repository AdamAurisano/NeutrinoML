'''
Activation layer that reimplements dense activation function as a sparse activation
'''

import MinkowskiEngine as ME

class MinkowskiActivation(ME.MinkowskiNetwork):

  def __init__(self, n_dims, a, **params):
    super(MinkowskiActivation, self).__init__(n_dims)
    self.a = a

  def forward(self, x):
    return ME.SparseTensor(self.a(x.F), coords_key=x.coords_key, coords_manager=x.coords_man)

