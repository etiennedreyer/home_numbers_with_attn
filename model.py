
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

def calc_output_size(module, input_size):

    current_height, current_width = input_size

    if isinstance(module, nn.Sequential):
        for layer in module:
            current_height, current_width = calc_output_size(layer, (current_height, current_width))

    if isinstance(module, nn.Conv2d):
        current_height = ((current_height - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0]) + 1
        current_width = ((current_width - module.kernel_size[1] + 2 * module.padding[1]) // module.stride[1]) + 1

    elif isinstance(module, nn.MaxPool2d):
        current_height = current_height // module.kernel_size
        current_width = current_width // module.kernel_size

    elif isinstance(module, nn.Linear):
        current_height = 1
        current_width = 1

    return current_height, current_width


# Adapted from https://github.com/Jongchan/attention-module
class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=(kernel_size-1) // 2, bias=True)

    def forward(self, x):
        channel_max      = torch.max(x,dim=1)[0].unsqueeze(1)
        channel_mean     = torch.mean(x,dim=1).unsqueeze(1)
        channel_mean_max = torch.cat( (channel_max,channel_mean), dim=1 )
        attn = self.conv(channel_mean_max)
        attn = F.sigmoid(attn)
        return x * attn

class AddressNet(nn.Module):

    def __init__(self,imshape,Ndigits=1,Nchannels=16,hidden_size=32,do_attention=True):
        super().__init__()
        
        C,H,W = imshape
        self.Ndigits = Ndigits
        self.Nchannels = Nchannels
        self.do_attention = do_attention

        if self.do_attention:
            self.SpatialGate = SpatialGate()

        self.f1 = nn.Sequential(OrderedDict([
            ('conv1',   nn.Conv2d(C,self.Nchannels,3,padding=0)),
            ('relu1',   nn.ReLU()),
            #('pool1',   nn.MaxPool2d(2,2)),
        ]))
        self.f1_out_shape = calc_output_size(self.f1,(H,W))

        self.f2 = nn.Sequential(OrderedDict([
            ('conv2',   nn.Conv2d(self.Nchannels,self.Nchannels,3,padding=0)),
            ('relu2',   nn.ReLU()),
            ('pool2',   nn.MaxPool2d(2,2)),
        ]))
        self.f2_out_shape = calc_output_size(self.f2,self.f1_out_shape)

        self.f3 = nn.Sequential(OrderedDict([
            ('conv3',   nn.Conv2d(self.Nchannels,self.Nchannels,3,padding=0)),
            ('relu3',   nn.ReLU()),
            #('pool3',   nn.MaxPool2d(2,2))
        ]))
        self.f3_out_shape = calc_output_size(self.f3,self.f2_out_shape)

        self.f4 = nn.Sequential(OrderedDict([
            ('conv3',   nn.Conv2d(self.Nchannels,self.Nchannels,3,padding=0)),
            ('relu3',   nn.ReLU()),
            ('pool3',   nn.MaxPool2d(2,2))
        ]))
        self.f4_out_shape = calc_output_size(self.f4,self.f3_out_shape)

        self.classifier = nn.Sequential(OrderedDict([
            ('full0', nn.Linear(self.Nchannels*self.f4_out_shape[0]*self.f4_out_shape[1], hidden_size)),
            ('relu5', nn.ReLU()),
            ('full1', nn.Linear(hidden_size,hidden_size//2)),
            ('relu6', nn.ReLU()),
            # ('full2', nn.Linear(hidden_size//2, 1)),
            ('full2', nn.Linear(hidden_size//2, self.Ndigits*10)),
        ]))

        
    def forward(self,x):

        if self.do_attention:
            x = self.SpatialGate(x)

        feats1 = self.f1(x)
        feats2 = self.f2(feats1)
        feats3 = self.f3(feats2)
        feats4 = self.f4(feats3)

        out = self.classifier(feats4.view(feats4.size(0), -1))

        return out