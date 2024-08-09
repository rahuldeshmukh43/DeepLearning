import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
from einops import rearrange
import math

class Conv_Layer(nn.Module):
    "conv + bn + relu"
    def __init__(self, 
                 in_channels:int,
                 out_channels: int,
                 kernel_size:int = 1,
                 stride: int = 1,                 
                 padding:int = 0 )-> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, 
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        return
    
    def forward(self, x):
        identity = x
        x = self.bn(self.conv(x))
        if self.stride == 1 and self.in_channels == self.out_channels:
            return identity + F.relu(x)
        else:
            return F.relu(x)

class SimpleCNN(nn.Module):
    """
    simple CNN for mnist dataset. returns a flattened vector embedding 
    """
    def __init__(self, 
                 in_channels:int = 1,
                 input_img_size:int = 32, # assuming square images
                 layer_config:List[int]= [3,3],
                 kernel_size:int = 3,
                 padding:int = 1,
                 out:int = 128
                 ) -> None:
        super().__init__()  
        self.in_channels = in_channels
        self.layer_config = layer_config
        self.kernel_size = kernel_size
        self.padding = padding
        self.out = out
        assert len(self.layer_config) == 2, "length of layer config should be 2"
        h = w = input_img_size
        # first layer 
        h = math.floor(1+ (h + 2*3 - 1*(7-1) - 1)/(1))
        w = math.floor(1+ (w + 2*3 - 1*(7-1) - 1)/(1))
        self.first_conv = Conv_Layer(in_channels,
                                    32,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3)   # [32, H, W]   
        h = math.floor(1+ (h + 2*1 - 1*(4-1) - 1)/(2))
        w = math.floor(1+ (w + 2*1 - 1*(4-1) - 1)/(2))
        self.first_pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1) #[32, H/2, W/2]
        # conv layers
        self.layers=  nn.ModuleList()
        out_channels = 32
        for i_stage, num_layers in enumerate(layer_config):
            for k in range(num_layers-1):
                # dont change out_channels
                h = math.floor(1+ (h + 2*padding - 1*(kernel_size-1) - 1)/(1))
                w = math.floor(1+ (w + 2*padding - 1*(kernel_size-1) - 1)/(1))
                self.layers.append(Conv_Layer(out_channels, out_channels,
                                              kernel_size=kernel_size, 
                                              padding=padding))

            # now use the stride as 2 and change out_channels
            h = math.floor(1+ (h + 2*padding - 1*(kernel_size-1) - 1)/(2))
            w = math.floor(1+ (w + 2*padding - 1*(kernel_size-1) - 1)/(2))
            self.layers.append(Conv_Layer(out_channels, 2*out_channels,
                                          kernel_size=kernel_size,
                                          stride=2,
                                          padding=padding))
            out_channels *= 2

        # linear layer
        # h = w = input_img_size // (2**(len(self.layer_config)+1))
        self.linear = nn.Linear(out_channels* h * w, out)

    def forward(self, x):
        """
        Args:
            x (Tensor) [B, C, H, W]
        """
        x = self.first_pool(self.first_conv(x)) #[B, C, H/2, W/2]

        for layer in self.layers:
            x = layer(x) 
        # [B, C * (2**(len(self.layer_config - 1 ))), H/8, W/8]
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.linear(x)
        return x


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.fc_layer = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten())
        self.out = 512
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.fc_layer(out)
        return out