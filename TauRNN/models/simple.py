'''Simple RNN network architecture for tau studies'''
import torch
from torch.nn import Linear, ReLU, RNN, LSTM, Softmax, Sequential as Seq

class TauRNN(torch.Module):
  def __init__(self, in_feats, hit_feats, point_feats,
    rnn_feats, hidden_feats, out_feats,
    lstm=False, dense_layers=5, rnn_layers=1):
    Super(self, SimpleRNN).__init__()
    rnn = LSTM if lstm else RNN
    self.hit_rnn = rnn(
      input_size=hit_feats,
      hidden_size=rnn_feats,
      num_layers=rnn_layers)
    self.point_rnn = rnn(
      input_size=point_feats,
      hidden_size=rnn_feats,
      num_layers=rnn_layers)
    dense = [ Linear(
      input_size=in_feats+2*rnn_feats,
      output_size=hidden_feats) ]
    for i in range(dense_layers-1):
      dense.append(Linear(
        input_size=hidden_feats,
        output_size=hidden_feats))
      dense.append(ReLU())
    dense.append(Softmax(-1))
    self.dense=Seq(*dense)

  def forward(self, x):
    print(x['x_fixed'].shape)
    print(x['x_point_var'].shape)
    print(x['x_hit_var'].shape)
    
