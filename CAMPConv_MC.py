# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:37:31 2023

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class conv3d_block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, 
                 stride=1, padding=1, dilation=1, groups=1,
                 relu=True, bn=True, bias=True):
        super(conv3d_block, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.1, 
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class conv2d_block(conv3d_block):
    def __init__(self, in_planes, out_planes, kernel_size, 
                 stride=1, padding=1, dilation=1, groups=1,
                 relu=True, bn=True, bias=True):
        super().__init__(in_planes, out_planes, kernel_size, stride,
                         padding, dilation, groups, relu, bn, bias)
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.1, 
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None 
        
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels=None, reduction_ratio=4, pool_types=['max']):
        '''
        Parameters
        ----------
        gate_channels : int, optional
            Number of filters in 3D convolution.
        reduction_ratio : int, optional
            Reduction ratio. The default is 4.
        pool_types : list, optional
            Pooling type. The default is ['max'].
        '''
        super(ChannelGate, self).__init__()
        self.planes = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(self.planes, self.planes // reduction_ratio),
            nn.ReLU(),
            nn.Linear(self.planes // reduction_ratio, self.planes)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        b, c, t, h, w = x.size()
        x = x.view(b, c, t, h, w)
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (t, h, w))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (t, h, w))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool3d(x, (t, h, w))
                channel_att_raw = self.mlp(lp_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum)
        scale = scale.view(b, c, 1, 1, 1)
        return x * scale + x   
    
class conv_channel_attention_3d(nn.Module):
    def __init__(self, reduction_ratio=4, period_num=4, filter_num=64, block_num=3):
        '''
        Parameters
        ----------
        reduction_ratio : int
            Reduction ratio of channel attention.
        period_num : int
            Sequence length of the input.
        filter_num : int
            Number of filters of 3D convolution.
        block_num : int
            Number of blocks used in the backbone.
        '''
        super(conv_channel_attention_3d, self).__init__()
        self.input = conv3d_block(3, filter_num, kernel_size=1, stride=1, padding=0)                    # input block
        self.net_3d = nn.ModuleList([conv3d_block(filter_num, filter_num, 3) for _ in range(block_num)])# backbone                                                                                                   
        self.output_3d = conv3d_block(filter_num, 1, kernel_size=1, stride=1, padding=0)                # output block 3d (compress channel dim)
        self.output_2d = conv2d_block(period_num, 1, kernel_size=1, stride=1, padding=0)                # output block 2d (compress temporal dim)
        self.attn_input = ChannelGate(filter_num, reduction_ratio=reduction_ratio)
        self.attns = nn.ModuleList([ChannelGate(filter_num, reduction_ratio=reduction_ratio) for _ in range(block_num)])
        
    def forward(self, x):
        out = self.attn_input(self.input(x)) # (B, 3, T, H, W) -> (B, C, T, H, W)
        for i, block in enumerate(self.net_3d): 
            out = self.attns[i](block(out)) # (B, C, T, H, W) -> (B, C, T, H, W)
        out = self.output_3d(out) # (B, C, T, H, W) -> (B, 1, T, H, W)
        out = out.squeeze(1) # (B, 1, T, H, W) -> (B, T, H, W)
        out = self.output_2d(out) # (B, T, H, W) -> (B, 1, H, W)
        out = out.unsqueeze(1) # (B, 1, H, W) -> (B, 1, 1, H, W)
        return out

if __name__ == '__main__':    
    x = torch.rand(size=(1, 3, 4, 42, 34)) # (B, C, T, H, W)
    model = conv_channel_attention_3d()
    pred = model(x)