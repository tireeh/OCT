# -*- coding: utf-8 -*-

"""
Created by Wang Han on 2018/7/4 15:31.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddCoords(nn.Module):
  """
  My implement of paper "An intriguing failing of convolutional neural networks
  and the CoordConv solution" whose address url: https://arxiv.org/pdf/1807.03247.pdf
  """

  def __init__(self, use_cuda=True, with_r=True):
    super(AddCoords, self).__init__()
    self.with_r = with_r
    self.use_cuda = use_cuda

  def forward(self, x):
    batch_size = x.shape[0]
    x_dim = x.shape[2]
    y_dim = x.shape[3]
    xx_ones = torch.ones([batch_size, x_dim])
    xx_ones = torch.unsqueeze(xx_ones, dim=-1)
    xx_range = torch.stack(
      batch_size * [torch.arange(start=0, end=y_dim, dtype=torch.float32)], 0)
    xx_range = torch.unsqueeze(xx_range, dim=1)
    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = torch.unsqueeze(xx_channel, dim=1)

    yy_ones = torch.ones([batch_size, y_dim])
    yy_ones = torch.unsqueeze(yy_ones, dim=1)
    yy_range = torch.stack(
      batch_size * [torch.arange(start=0, end=x_dim, dtype=torch.float32)], 0)
    yy_range = torch.unsqueeze(yy_range, dim=-1)
    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = torch.unsqueeze(yy_channel, dim=1)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    if self.use_cuda:
      xx_channel = xx_channel.cuda()
      yy_channel = yy_channel.cuda()
    ret = torch.cat([x, xx_channel, yy_channel], dim=1)
    if self.with_r:
      rr = torch.sqrt(torch.pow(xx_channel, 2) +
                      torch.pow(yy_channel, 2))
      if self.use_cuda:
        rr = rr.cuda()
      ret = torch.cat([ret, rr], dim=1)
    return ret


class CoordConv2d(nn.Module):
  def __init__(self, use_cuda=True, with_r=False, *args, **kwargs):
    super(CoordConv2d, self).__init__()
    self.addcoords = AddCoords(use_cuda=use_cuda, with_r=with_r)
    self.conv = nn.Conv2d(*args, **kwargs)

  def forward(self, x):
    ret = self.addcoords(x)
    ret = self.conv(ret)
    return ret


class CoordConvTranspose2d(nn.Module):
  def __init__(self, with_r=False, *args, **kwargs):
    super(CoordConvTranspose2d, self).__init__()
    self.addcoords = AddCoords(with_r=with_r)
    self.conv = nn.ConvTranspose2d(*args, **kwargs)

  def forward(self, x):
    ret = self.addcoords(x)
    ret = self.conv(ret)
    return ret


class DoubleConv(nn.Module):
  """(conv => BN => ReLU) * 2"""

  def __init__(self, in_ch, out_ch):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
      CoordConv2d(in_channels=in_ch + 2, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
      CoordConv2d(in_channels=out_ch + 2, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
    )

  def forward(self, x):
    x = self.conv(x)
    return x


class InConv(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(InConv, self).__init__()
    self.conv = DoubleConv(in_ch, out_ch)

  def forward(self, x):
    x = self.conv(x)
    return x


class DownConv(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(DownConv, self).__init__()
    self.conv = nn.Sequential(
      nn.MaxPool2d(2),
      DoubleConv(in_ch, out_ch)
    )

  def forward(self, x):
    x = self.conv(x)
    return x


class UpConv(nn.Module):
  def __init__(self, in_ch, out_ch, bilinear=False):
    super(UpConv, self).__init__()

    if bilinear:
      self.up = nn.Upsample(
        scale_factor=2, mode='bilinear', align_corners=True)
    else:
      self.up = CoordConvTranspose2d(in_channels=in_ch // 2 + 2, out_channels=in_ch // 2, kernel_size=2, stride=2)

    self.conv = DoubleConv(in_ch, out_ch)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    diff_x = x2.size()[2] - x1.size()[2]
    diff_y = x2.size()[3] - x1.size()[3]
    pad_x_l, pad_y_l = diff_x // 2, diff_y // 2
    pdd_x_r, pad_y_r = diff_x - pad_x_l, diff_y - pad_y_l
    x1 = F.pad(x1, (pad_y_l, pad_y_r, pad_x_l, pdd_x_r))
    x = torch.cat([x2, x1], dim=1)
    x = self.conv(x)
    return x


class OutConv(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(OutConv, self).__init__()
    self.conv = CoordConv2d(in_channels=in_ch + 2, out_channels=out_ch, kernel_size=1)

  def forward(self, x):
    x = self.conv(x)
    return x


class UNet_coordconv(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(UNet_coordconv, self).__init__()
    self.inconv = InConv(n_channels, 32)
    self.down1 = DownConv(32, 64)
    self.down2 = DownConv(64, 128)
    self.down3 = DownConv(128, 256)
    self.down4 = DownConv(256, 256)
    self.up1 = UpConv(512, 128)
    self.up2 = UpConv(256, 64)
    self.up3 = UpConv(128, 32)
    self.up4 = UpConv(64, 32)
    self.outconv = OutConv(32, n_classes)

    self._initialize_weights()

  def forward(self, x):
    x1 = self.inconv(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = self.outconv(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
