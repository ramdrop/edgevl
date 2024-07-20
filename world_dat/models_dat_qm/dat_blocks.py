# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_

from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn
from quantization_libs.quantizer import QWrap_Conv2d, QWarp_Linear, Quantizer



class LocalAttention(nn.Module):
    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, **kwargs):

        super().__init__()
        quant_desc_input = QuantDescriptor(num_bits=quant_bit, calib_method='histogram')
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        window_size = to_2tuple(window_size)

        self.proj_qkv = quant_nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.proj_out = quant_nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads))
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))                  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)                                                  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]                  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()                            # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1                                        # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                                          # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # quantization step:
        self.q_des_i = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=None, calib_method=kwargs['quantization']['act_cal']['calib_method'])
        self.q_des_w = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=(0) if kwargs['quantization']['weight_t_c'] == 'per_channel' else None, calib_method=kwargs['quantization']['weight_cal']['calib_method'])

        self.q_w_q = TensorQuantizer(self.q_des_i)
        self.q_w_k = TensorQuantizer(self.q_des_i)
        self.q_w_v = TensorQuantizer(self.q_des_i)
        self.q_atten = TensorQuantizer(self.q_des_i)
        self.q_bias = TensorQuantizer(self.q_des_w)
        self.q_atten1 = TensorQuantizer(self.q_des_i)

    def forward(self, x, mask=None):

        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0], w1=self.window_size[1])        # B x Nr x Ws x C
        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')

        qkv = self.proj_qkv(x_total)             # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)
        q = q * self.scale
        q = self.q_w_q(q) # quantization q k v
        k = self.q_w_k(k)
        v = self.q_w_v(v)

        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()                                                                                                                    # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn_bias = self.q_bias(attn_bias)
        attn = self.q_atten(attn)
        attn = attn + attn_bias.unsqueeze(0)
        attn = self.q_atten1(attn)

        if mask is not None:
            # attn : (b * nW) h w w
            # mask : nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww, w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))
        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x))                                                                                                         # B' x N x C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1])          # B x C x H x W
        return x, None, None


class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size, **kwargs):

        super().__init__(dim, heads, window_size, attn_drop, proj_drop)
        # quant_desc_input = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=None, calib_method=kwargs['quantization']['calib_method'])  # per-tensor, max
        # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size)                                                     # H W
        h_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0], w1=self.window_size[1]) # nW, ww
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)                          # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x, None, None


class DAttentionBaseline(nn.Module):

    def __init__(self, q_size, kv_size, n_heads, n_head_channels, n_groups, attn_drop, proj_drop, stride, offset_range_factor, use_pe, dwc_pe, no_off, fixed_pe, ksize, log_cpb, **kwargs):

        super().__init__()
        # quant_desc_input = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=None, calib_method=kwargs['quantization']['act_cal']['calib_method'])  # per-tensor, max
        # quant_desc_weight = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=(0) if kwargs['quantization']['weight_t_c'] == 'per_channel' else None, calib_method=kwargs['quantization']['weight_cal']['calib_method'])  # per-tensor or per-channel, max
        # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        # quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        # quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels**-0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            # quant_nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.n_group_channels, out_channels=self.n_group_channels, kernel_size=kk, stride=stride, padding=pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            # quant_nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.n_group_channels, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False),
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # self.proj_q = quant_nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_q = QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.nc, out_channels=self.nc, kernel_size=1, stride=1, padding=0)
        # self.proj_k = quant_nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.nc, out_channels=self.nc, kernel_size=1, stride=1, padding=0)
        # self.proj_v = quant_nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.nc, out_channels=self.nc, kernel_size=1, stride=1, padding=0)
        # self.proj_out = quant_nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.nc, out_channels=self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                # self.rpe_table = quant_nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
                self.rpe_table = QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.nc, out_channels=self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w))
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                # self.rpe_table = nn.Sequential(quant_nn.Linear(2, 32, bias=True), nn.ReLU(inplace=True), quant_nn.Linear(32, self.n_group_heads, bias=False))
                self.rpe_table = nn.Sequential(
                    QWarp_Linear(quant_config=kwargs['quantization'], in_features=2, out_features=32, bias=True),
                    nn.ReLU(inplace=True),
                    QWarp_Linear(quant_config=kwargs['quantization'], in_features=32, out_features=self.n_group_heads, bias=False),
                )
            else:
                self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1))
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
        # quantization step:
        # self.q_des_w = QuantDescriptor(num_bits=quant_bit, axis=(0))

        # self.q_des_t = QuantDescriptor(num_bits=kwargs['quantization']['bit'], axis=None, calib_method=kwargs['quantization']['act_cal']['calib_method'])  # per-tensor, max
        # self.q_ref = TensorQuantizer(self.q_des_t)
        self.q_ref = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_offset = TensorQuantizer(self.q_des_t)
        self.q_offset = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_w_q = TensorQuantizer(self.q_des_t)
        self.q_w_q = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_w_k = TensorQuantizer(self.q_des_t)
        self.q_w_k = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_w_v = TensorQuantizer(self.q_des_t)
        self.q_w_v = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_atten = TensorQuantizer(self.q_des_t)
        self.q_atten = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_bias  = TensorQuantizer(self.q_des_t)
        self.q_bias = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_atten1 = TensorQuantizer(self.q_des_t)
        self.q_atten1 = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')
        # self.q_atten2 = TensorQuantizer(self.q_des_t)
        self.q_atten2 = Quantizer(quant_config=kwargs['quantization'], neuro_type='activation')

        # self.q_residue = TensorQuantizer(self.q_des_w)
        # self.q_atten3 = TensorQuantizer(self.q_des_w)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device), indexing='ij')
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)                                 # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(torch.arange(0, H, dtype=dtype, device=device), torch.arange(0, W, dtype=dtype, device=device), indexing='ij')
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)                                 # B * g H W 2

        return ref

    def forward(self, x):                        # ([128, 64, 56, 56])

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)                       # ([128, 64, 56, 56])
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)            # ([128, 64, 56, 56])
        offset = self.conv_offset(q_off).contiguous()                                              # B * g 2 Hg Wg  ([128, 2, 7, 7])
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)                                 # ([128, 7, 7, 2])
        offset = self.q_offset(offset)
        reference = self.q_ref(reference)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)                                             # ([128, 7, 7, 2])

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),                   # ([128, 64, 56, 56])
                grid=pos[..., (1, 0)],                                                             # y, x -> x, y
                mode='bilinear',
                align_corners=True)                                                                # B * g, Cg, Hg, Wg ([128, 64, 7, 7])

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)                                           # ([128, 64, 1, 49])

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)                               # ([256, 32, 3136])
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)       # ([256, 32, 49])
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)       # ([256, 32, 49])
        q = self.q_w_q(q)
        k = self.q_w_k(k)
        v = self.q_w_v(v)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)                                         # B * h, HW, Ns ([256, 3136, 49])
        attn = attn.mul(self.scale)
        self.q_atten(attn)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
                # residual_lepe = self.q_residue(residual_lepe)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn_bias = self.q_bias(attn_bias)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = self.q_atten1(attn)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(4.0)                                                         # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)
                attn_bias = self.q_bias(attn_bias)                                                                                                                                                              # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_gorup_heads)
                attn = self.q_atten1(attn)
            else:
                rpe_table = self.rpe_table       # ([2, 111, 111])
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)                              # ([128, 2, 111, 111])
                q_grid = self._get_q_grid(H, W, B, dtype, device)                                  # ([128, 56, 56, 2])
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5) # ([128, 3136, 49, 2]) <= ([128, 3136, 1, 2]) - ([128, 1, 49, 2])
                attn_bias = F.grid_sample(input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups), grid=displacement[..., (1, 0)], mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns ([128, 2, 3136, 49])
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn_bias = self.q_bias(attn_bias)
                attn = attn + attn_bias
                attn = self.q_atten1(attn)

        attn = F.softmax(attn, dim=2)            # ([256, 3136, 49])
        attn = self.attn_drop(attn)
        attn = self.q_atten2(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)                                       # ([256, 32, 3136])

        if self.use_pe and self.dwc_pe:
            # self.q_atten3(out)
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)            # ([128, 64, 56, 56])

        y = self.proj_drop(self.proj_out(out))   # ([128, 64, 56, 56])

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class PyramidAttention(nn.Module):

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):

        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = quant_nn.Conv2d(dim, dim, 1, 1, 0)
        self.kv = quant_nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = quant_nn.Conv2d(dim, dim, 1, 1, 0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.proj_ds = nn.Sequential(quant_nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio), LayerNormProxy(dim))
        # self.q_des_w = QuantDescriptor(num_bits=8, axis=(0), calib_method='histogram')
        self.q_des_i = QuantDescriptor(num_bits=8, calib_method='histogram')
        self.q_w_q = TensorQuantizer(self.q_des_i)
        self.q_w_k = TensorQuantizer(self.q_des_i)
        self.q_w_v = TensorQuantizer(self.q_des_i)
        self.q_atten = TensorQuantizer(self.q_des_i)
        self.q_atten1 = TensorQuantizer(self.q_des_i)
    def forward(self, x):

        B, C, H, W = x.size()
        Nq = H * W
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ds = self.proj_ds(x)
            kv = self.kv(x_ds)
        else:
            kv = self.kv(x)

        k, v = torch.chunk(kv, 2, dim=1)
        Nk = (H // self.sr_ratio) * (W // self.sr_ratio)
        q = q.reshape(B * self.num_heads, self.head_dim, Nq).mul(self.scale)
        k = k.reshape(B * self.num_heads, self.head_dim, Nk)
        v = v.reshape(B * self.num_heads, self.head_dim, Nk)
        q = self.q_w_q(q)
        k = self.q_w_k(k)
        v = self.q_w_v(v)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = self.q_atten(attn)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        attn = self.q_atten1(attn)
        x = torch.einsum('b m n, b c n -> b c m', attn, v)
        x = x.reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None, None


class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop, **kwargs):

        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        # self.chunk.add_module('linear1', quant_nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('linear1', QWarp_Linear(quant_config=kwargs['quantization'], in_features=self.dim1, out_features=self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        # self.chunk.add_module('linear2', quant_nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('linear2', QWarp_Linear(quant_config=kwargs['quantization'], in_features=self.dim2, out_features=self.dim1))
        # self.chunk.add_module('drop2', quant_nn.Dropout(drop, inplace=True))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))

    def forward(self, x):

        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class LayerNormProxy(nn.Module):

    def __init__(self, dim):

        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop, **kwargs):

        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            # quant_nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            QWrap_Conv2d(quant_config=kwargs['quantization'],  in_channels=self.dim1, out_channels=self.dim2, kernel_size=1, stride=1, padding=0),
            # nn.GELU(),
            # nn.BatchNorm2d(self.dim2, eps=1e-5)
        )
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        # self.bn = nn.BatchNorm2d(self.dim2, eps=1e-5)
        self.linear2 = nn.Sequential(
            # quant_nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),\
            QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.dim2, out_channels=self.dim1, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(self.dim1, eps=1e-5)
        )
        self.drop2 = nn.Dropout(drop, inplace=True)
        # self.dwc = quant_nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
        self.dwc = QWrap_Conv2d(quant_config=kwargs['quantization'], in_channels=self.dim2, out_channels=self.dim2, kernel_size=3, stride=1, padding=1, groups=self.dim2)

    def forward(self, x):

        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        # x = self.bn(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x
