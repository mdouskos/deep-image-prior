import torch
import torch.nn as nn
from .common import *

def decoder(
        num_input_channels=2,
        num_output_channels=3, 
        num_channels_up=[16, 32, 64, 128, 128], 
        filter_size_up=3, 
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles decoder.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    
    n_scales = len(num_channels_up) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales
   
    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    for i in range(len(num_channels_up)):
        block = nn.Sequential()
        
        if i == 0:
            # The deepest
            k = num_input_channels
        else:
            k = num_channels_up[i - 1]

        block.add(conv(k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        block.add(bn(num_channels_up[i]))
        block.add(act(act_fun))


        if need1x1_up:
            block.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            block.add(bn(num_channels_up[i]))
            block.add(act(act_fun))

        model.add(block)


        model.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

    model.add(conv(num_channels_up[-1], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
