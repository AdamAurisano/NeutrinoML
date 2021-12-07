"""
Module for holding onto PyTorch Geometric network architectures
"""

def get_model(name, **model_args):
  if name == "message":
    from .message_passing import GNNSegmentClassifier
    return GNNSegmentClassifier(**model_args)
  elif name == "DeepMultiHead":
    from .message_passing_multihead_deep import GNNDeepMultiHead
    return GNNDeepMultiHead(**model_args)
  elif name == "3DeepMultiHead":
    from .message_passing_multihead_3deep import GNN3DeepMultiHead
    return GNN3DeepMultiHead(**model_args)
  else:
    raise Exception(f"Model {name} unknown.")

