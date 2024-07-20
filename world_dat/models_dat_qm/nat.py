#################################################################################################
# Copyright (c) 2023 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# Originated from https://github.com/SHI-Labs/NATTEN/blob/main/src/natten/natten2d.py
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn
from quantization_libs.quantizer import QWrap_Conv2d, QWarp_Linear, Quantizer


class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """
    def __init__(self, dim, kernel_size, num_heads, attn_drop=0., proj_drop=0.,
                 dilation=None, **kwargs):
        super().__init__()
        # quant_desc_input = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=None, calib_method=kwargs['quantization']['act_cal']['calib_method'])  # per-tensor, max
        # quant_desc_weight = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=(0) if kwargs['quantization']['weight_t_c'] == 'per_channel' else None, calib_method=kwargs['quantization']['weight_cal']['calib_method'])
        # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        # quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        # quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
        self.fp16_enabled = False
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size
        if type(dilation) is str:
            self.dilation = None
            self.window_size = None
        else:
            assert dilation is None or dilation >= 1, \
                f"Dilation must be greater than or equal to 1, got {dilation}."
            self.dilation = dilation or 1
            self.window_size = self.kernel_size * self.dilation

        # self.qkv = quant_nn.Linear(dim, dim * 3)
        self.qkv = QWarp_Linear(quant_config=kwargs['quantization'], in_features=dim, out_features=dim * 3)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = quant_nn.Linear(dim, dim)
        self.proj = QWarp_Linear(quant_config=kwargs['quantization'], in_features=dim, out_features=dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #quantization
        # self.q_des_i = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=None, calib_method=kwargs['quantization']['act_cal']['calib_method'])  # per-tensor, max, activation
        # self.q_w_q = TensorQuantizer(self.q_des_i)
        self.q_w_q = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_w_k = TensorQuantizer(self.q_des_i)
        self.q_w_k = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_w_v = TensorQuantizer(self.q_des_i)
        self.q_w_v = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_atten = TensorQuantizer(self.q_des_i)
        self.q_atten = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')

    def forward(self, x): # ([128, 64, 56, 56])
        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        x = x.permute(0, 2, 3, 1)                # ([128, 56, 56, 64])
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        dilation = self.dilation
        window_size = self.window_size
        if window_size is None:
            dilation = max(min(H, W) // self.kernel_size, 1)
            window_size = dilation * self.kernel_size
        if H < window_size or W < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - W)
            pad_b = max(0, window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape

        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)                 # ([128, 56, 56, 192]).resh.. => ([3, 128, 2, 56, 56, 32])
        q, k, v = qkv[0], qkv[1], qkv[2]         # ([128, 2, 56, 56, 32]) *s3
        #quantize q k v
        q = self.q_w_q(q)
        k = self.q_w_k(k)
        v = self.q_w_v(v)
        q = q * self.scale
        # breakpoint()
        attn = NATTEN2DQKRPBFunction.apply(q, k, self.rpb, self.kernel_size, dilation)             # self.rpb:([2, 13, 13]), ([128, 2, 56, 56, 49])
        # quantize attn
        attn = self.q_atten(attn)
        attn = attn.softmax(dim=-1)              # ([128, 2, 56, 56, 49])
        attn = self.attn_drop(attn)
        x = NATTEN2DAVFunction.apply(attn, v, self.kernel_size, dilation)                          # ([128, 2, 56, 56, 32])
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)                                           # ([128, 56, 56, 64])
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2), None, None
