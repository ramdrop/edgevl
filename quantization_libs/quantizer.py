import sys
sys.path.append('.')
sys.path.append('..')

import torch.nn as nn

# Pytorch-Quantization Library
import pytorch_quantization.nn as pq_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

# LSQ Library
import quantization_libs.lsq_quant as lsq_quant
from quantization_libs.lsq_quant.quantizer.lsq import LsqQuan
# from quantization_libs.lsq_quant.func import QuanConv2d, QuanLinear

# EWGS Library
import quantization_libs.ewgs_quant as ewgs_quant
from quantization_libs.ewgs_quant.ewgs import QFDQuantizer


def to_config_lsq(quant_config):
    lc = {}
    lc['bit'] = quant_config['num_bits']
    lc['all_positive'] = False
    lc['symmetric'] = True
    lc['per_channel'] = True if quant_config['axis'] is not None else False
    return lc

def to_config_qfd(quant_config):
    lc = {}
    lc['num_levels'] = 2**quant_config['num_bits']
    return lc

def to_config_jacob(quant_config):
    config = {}
    config['num_bits'] = quant_config['num_bits']
    config['calib_method'] = quant_config['calib_method']

    remap = {'per_tensor': None, 'per_channel': 0}
    assert quant_config['axis'] in remap, f"Invalid axis: {quant_config['axis']}"
    config['axis'] = remap[quant_config['axis']]

    return config


def Quantizer(quant_config, neuro_type):
    if quant_config['method'] == 'disable':
        return nn.Identity()
    elif quant_config['method'] == 'jacob':
        quant_desc = QuantDescriptor(**to_config_jacob(quant_config[neuro_type]))
        neuro_quantizer = TensorQuantizer(quant_desc)
        return neuro_quantizer
    elif quant_config['method'] == 'lsq':
        return LsqQuan(**to_config_lsq(quant_config[neuro_type]))
    elif quant_config['method'] == 'qfd':
        return QFDQuantizer(**to_config_qfd(quant_config[neuro_type]))
    else:
        raise NotImplementedError


def QWrap_Conv2d(quant_config, **kwargs):
    if quant_config['method'] == 'disable':
        return nn.Conv2d(**kwargs)
    elif quant_config['method'] == 'jacob':
        quant_desc_input = QuantDescriptor(**to_config_jacob(quant_config['activation'])) # per-tensor, max
        quant_desc_weight = QuantDescriptor(**to_config_jacob(quant_config['weight']))
        kwargs['quant_desc_input'] = quant_desc_input
        kwargs['quant_desc_weight'] = quant_desc_weight
        return pq_nn.QuantConv2d(**kwargs)
    elif quant_config['method'] == 'lsq':
        w_quantizer = LsqQuan(**to_config_lsq(quant_config['weight']))
        a_quantizer = LsqQuan(**to_config_lsq(quant_config['activation']))
        return lsq_quant.QuanConv2d(nn.Conv2d(**kwargs), quan_w_fn=w_quantizer, quan_a_fn=a_quantizer)
    elif quant_config['method'] == 'qfd':
        w_quantizer = QFDQuantizer(is_weight=True, **to_config_qfd(quant_config['weight']))
        a_quantizer = QFDQuantizer(**to_config_qfd(quant_config['activation']))
        return ewgs_quant.QuanConv2d(nn.Conv2d(**kwargs), quan_w_fn=w_quantizer, quan_a_fn=a_quantizer)
    else:
        raise NotImplementedError


def QWarp_Linear(quant_config, **kwargs):
    if quant_config['method'] == 'disable':
        return nn.Linear(**kwargs)
    elif quant_config['method'] == 'jacob':
        quant_desc_input = QuantDescriptor(**to_config_jacob(quant_config['activation'])) # per-tensor, max
        quant_desc_weight = QuantDescriptor(**to_config_jacob(quant_config['weight']))
        kwargs['quant_desc_input'] = quant_desc_input
        kwargs['quant_desc_weight'] = quant_desc_weight
        return pq_nn.QuantLinear(**kwargs)
    elif quant_config['method'] == 'lsq':
        w_quantizer = LsqQuan(**to_config_lsq(quant_config['weight']))
        a_quantizer = LsqQuan(**to_config_lsq(quant_config['activation']))
        return lsq_quant.QuanLinear(nn.Linear(**kwargs), quan_w_fn=w_quantizer, quan_a_fn=a_quantizer)
    elif quant_config['method'] == 'qfd':
        w_quantizer = QFDQuantizer(**to_config_qfd(quant_config['weight']))
        a_quantizer = QFDQuantizer(**to_config_qfd(quant_config['activation']))
        return ewgs_quant.QuanLinear(nn.Linear(**kwargs), quan_w_fn=w_quantizer, quan_a_fn=a_quantizer)
    else:
        raise NotImplementedError


from .linear_activation import QuantizedConv2d, LinearActivation

def QWrap_Conv2d_Act(quant_config, **kwargs):
    if quant_config['method'] == 'disable':
        return nn.Conv2d(**kwargs)

    elif quant_config['method'] == 'jacob':
        quant_desc_input = QuantDescriptor(**to_config_jacob(quant_config['activation']))
        quant_desc_weight = QuantDescriptor(**to_config_jacob(quant_config['weight']))
        kwargs['quant_desc_input'] = quant_desc_input
        kwargs['quant_desc_weight'] = quant_desc_weight
        return QuantizedConv2d(**kwargs)

    elif quant_config['method'] == 'lsq':
        w_quantizer = LsqQuan(**to_config_lsq(quant_config['weight']))
        a_quantizer = LsqQuan(**to_config_lsq(quant_config['activation']))
        return lsq_quant.QuanConv2d(nn.Conv2d(**kwargs), quan_w_fn=w_quantizer, quan_a_fn=a_quantizer)

    elif quant_config['method'] == 'qfd':
        w_quantizer = QFDQuantizer(is_weight=True, **to_config_qfd(quant_config['weight']))
        a_quantizer = QFDQuantizer(**to_config_qfd(quant_config['activation']))
        return ewgs_quant.QuanConv2d(nn.Conv2d(**kwargs), quan_w_fn=w_quantizer, quan_a_fn=a_quantizer)
    else:
        raise NotImplementedError


def QWarp_Linear_Act(quant_config, **kwargs):
    if quant_config['method'] == 'jacob':
        quant_desc_input = QuantDescriptor(**to_config_jacob(quant_config['activation'])) # per-tensor, max
        quant_desc_weight = QuantDescriptor(**to_config_jacob(quant_config['weight']))
        kwargs['quant_desc_input'] = quant_desc_input
        kwargs['quant_desc_weight'] = quant_desc_weight
        return LinearActivation(**kwargs)
    else:
        # remove key act
        if 'act' in kwargs:
            kwargs.pop('act')
        if quant_config['method'] == 'disable':
            return nn.Linear(**kwargs)
        elif quant_config['method'] == 'lsq':
            w_quantizer = LsqQuan(**to_config_lsq(quant_config['weight']))
            a_quantizer = LsqQuan(**to_config_lsq(quant_config['activation']))
            return lsq_quant.QuanLinear(nn.Linear(**kwargs), quan_w_fn=w_quantizer, quan_a_fn=a_quantizer)
        elif quant_config['method'] == 'qfd':
            w_quantizer = QFDQuantizer(**to_config_qfd(quant_config['weight']))
            a_quantizer = QFDQuantizer(**to_config_qfd(quant_config['activation']))
            return ewgs_quant.QuanLinear(nn.Linear(**kwargs), quan_w_fn=w_quantizer, quan_a_fn=a_quantizer)
        else:
            raise NotImplementedError
