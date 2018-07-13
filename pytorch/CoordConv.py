'''
From An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution (http://arxiv.org/abs/1807.03247)
Translated to pytorch.
'''

import torch
import torch.nn as nn


class AddCoords(torch.nn.Module):
    """Add coords to a tensor"""
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
    
    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        input_tensor = input_tensor.permute(0, 2, 3, 1)
        batch_size_tensor = input_tensor.shape[0]
        xx_ones = torch.ones([batch_size_tensor, self.x_dim],
                         dtype=torch.int32)
        xx_ones = torch.unsqueeze(xx_ones, -1)
        xx_range = torch.unsqueeze(torch.arange(self.x_dim, dtype=torch.int32), dim=0).repeat(batch_size_tensor, 1)
        xx_range = torch.unsqueeze(xx_range, 1)
        
        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = torch.unsqueeze(xx_channel, -1)
        
        yy_ones = torch.ones([batch_size_tensor, self.y_dim],
                         dtype=torch.int32)
        yy_ones = torch.unsqueeze(yy_ones, 1)
        yy_range = torch.unsqueeze(torch.arange(self.y_dim, dtype=torch.int32), dim=0).repeat(batch_size_tensor, 1)
        yy_range = torch.unsqueeze(yy_range, -1)
        
        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = torch.unsqueeze(yy_channel, -1)
        
        xx_channel = xx_channel.float() / (self.x_dim - 1)
        yy_channel = yy_channel.float() / (self.y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        
        ret = torch.cat((input_tensor,
                        xx_channel,
                        yy_channel), dim=-1)
        
        if self.with_r:
            rr = torch.sqrt(torch.mul(xx_channel - 0.5, xx_channel - 0.5)
                           + torch.mul(yy_channel - 0.5, yy_channel - 0.5))
            ret = torch.cat((ret, rr), dim=-1)
        
        ret = ret.permute(0, 3, 1, 2)
        return ret

    
class CoordConv(torch.nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, in_channels, out_channels, kernel_size, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim,
                                  y_dim=y_dim,
                                  with_r=with_r)
        self.conv = nn.Conv2d(in_channels+2+with_r, out_channels, kernel_size, *args, **kwargs)
        
    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret

