'''
Module for retrieving PyTorch network architectures
'''

def get_model(name, **model_args):
    
    if name == 'MinkowskiSeg':
        from .minkowski_seg import MinkowskiSeg
        return MinkowskiSeg(**model_args)
    elif name == 'MinkowskiASPPSeg':
        from .minkowski_ASPPSeg import MinkowskiASPPSeg
        return MinkowskASPPSeg(**model_args)
    elif name == 'PanopticSeg':
        from .minkowski_panoptic_seg import PanopticSeg
        return PanopticSeg(**model_args)
    elif name == 'MinkowskiClass':
        from .minkowski_class import MinkowskiClass
        return MinkowskiClass(**model_args)
    elif name == 'Minkowski2StackClass':
        from .minkowski_class import Minkowski2StackClass
        return Minkowski2StackClass(**model_args)
    elif name == 'MobileNet':
        from .nova_mobilenet import MobileNet
        return MobileNet(**model_args)
    elif name == 'MobileNetNoUnion':
        from .nova_mobilenet_nounion import MobileNetNoUnion
        return MobileNetNoUnion(**model_args)
    elif name == 'DenseMobileNet':
        from .nova_dense_mobilenet import DenseMobileNet
        return DenseMobileNet(**model_args)
    elif name == 'MultiHead':
        from .message_passing_multihead import MultiHead
        return MultiHead(**model_args)
    elif name == 'DeepMultiHead':
        from .message_passing_multihead_deep import GNNDeepMultiHead
        return GNNDeepMultiHead(**model_args)
    elif name == 'FishNet':
        from .fishnet import fish
        return fish(**model_args)
    else:
        raise Exception(f'Model {name} unknown.')
