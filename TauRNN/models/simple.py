'''Simple RNN network architecture for tau studies'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

class TauRNN(nn.Module):
  def __init__(self, in_feats, hit_feats, point_feats,
    rnn_feats, hidden_feats, out_feats,
    lstm=False, dense_layers=5, rnn_layers=1):
    super(TauRNN, self).__init__()
    rnn = nn.LSTM if lstm else nn.RNN
    self.hit_rnn = rnn(
      input_size=hit_feats,
      hidden_size=rnn_feats,
      num_layers=rnn_layers,
      batch_first=True)
    self.point_rnn = rnn(
      input_size=point_feats,
      hidden_size=rnn_feats,
      num_layers=rnn_layers,
      batch_first=True)
    dense = [ nn.Linear(
      in_features=in_feats+2*rnn_feats,
      out_features=hidden_feats) ]
    for i in range(dense_layers-1):
      dense.append(nn.Linear(
        in_features=hidden_feats, 
        out_features=hidden_feats))
      dense.append(nn.ReLU())
    dense.append(nn.Linear(
      in_features=hidden_feats,
      out_features=out_feats))
    dense.append(nn.Sigmoid())
    self.dense=nn.Sequential(*dense)

  def forward(self, x):
    
    x_point, _ = self.point_rnn(x[1])
    x_point, length = pad_packed_sequence(x_point)
    x_point = torch.stack([ x_point[j-1, i, :] for i, j in enumerate(length) ])
    x_hit, _ = self.hit_rnn(x[2])
    x_hit, length = pad_packed_sequence(x_hit)
    x_hit = torch.stack([ x_hit[j-1, i, :] for i, j in enumerate(length) ])
    x = torch.cat([x[0], x_point, x_hit], dim=1)
    return self.dense(x).squeeze(dim=-1)
