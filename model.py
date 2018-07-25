#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yang
"""

import torch.nn as nn 
import torch
import torch.nn.functional as F
import numpy as np

class WavenetAutoencoder(nn.Module):

    def __init__(self,

        split_gpus,
        filter_width,
        quantization_channel,
        dilations,
         
        en_residual_channel,
        en_dilation_channel,
       
        en_bottleneck_width,
        en_pool_kernel_size,

       
        de_residual_channel,
        de_dilation_channel,
        de_skip_channel,

        use_bias):

        super(WavenetAutoencoder,self).__init__()

        self.filter_width = filter_width
        self.quantization_channel = quantization_channel
        self.dilations = dilations

        self.en_residual_channel = en_residual_channel
        self.en_dilation_channel = en_dilation_channel
       
        self.en_bottleneck_width = en_bottleneck_width
        self.en_pool_kernel_size = en_pool_kernel_size
        
       
        self.de_residual_channel = de_residual_channel
        self.de_dilation_channel = de_dilation_channel
        self.de_skip_channel = de_skip_channel

        self.use_bias = use_bias
        self.cuda_available = torch.cuda.is_available()

        self.receptive_field = (self.filter_width - 1) * (sum(self.dilations) + 1) + 1
        
        self.decoder = Decoder(self.filter_width, self.quantization_channel, self.dilations, self.en_bottleneck_width,
                               self.de_residual_channel, self.de_dilation_channel, self.de_skip_channel,
                               self.use_bias, self.cuda_available, self.receptive_field)
        self.encoder = Encoder(self.filter_width, self.quantization_channel, self.dilations, self.en_bottleneck_width,
                               self.en_residual_channel, self.en_dilation_channel, self.en_pool_kernel_size, self.use_bias)
        
        self.split_gpus = split_gpus
        if self.split_gpus:
            self.encoder.cuda(0)
            self.decoder.cuda(1)

    def forward(self,wave_input):
        print("input size = " , wave_input.data.size())
        batch_size, original_channels, seq_len = wave_input.size()

        output_width = seq_len - self.receptive_field + 1
        encoding = self.encoder(wave_input)
        print(" encoding size = " , encoding.data.size())
       # encoding = encoding.cuda()
        if self.split_gpus:
            encoding = encoding.cuda(1)
        result = self.decoder(wave_input,encoding,output_width)
        print(" output size = " , result.data.size())
        return result

class Encoder(nn.Module):

    def __init__(self, 

        filter_width,
        quantization_channel,
        dilations,
                 
        en_bottleneck_width,
        en_residual_channel,
        en_dilation_channel,
        en_pool_kernel_size,

        use_bias):

        super(Encoder, self).__init__()

        self.filter_width = filter_width
        self.quantization_channel = quantization_channel
        self.dilations = dilations

        self.en_residual_channel = en_residual_channel
        self.en_dilation_channel = en_dilation_channel

        self.en_bottleneck_width = en_bottleneck_width
        self.en_pool_kernel_size = en_pool_kernel_size
        self.use_bias = use_bias

        self.en_pool_kernel_size = en_pool_kernel_size
        self.pool1d = nn.AvgPool1d(self.en_pool_kernel_size)

        self.bottleneck_layer = nn.Conv1d(self.en_residual_channel, self.en_bottleneck_width, 1,bias=self.use_bias)
        self.en_causal_layer = nn.Conv1d(self.quantization_channel, self.en_residual_channel, self.filter_width,bias = self.use_bias)

        self.en_dilation_layer_stack = nn.ModuleList()
        self.en_dense_layer_stack = nn.ModuleList()

        for dilation in self.dilations:

            self.en_dilation_layer_stack.append( nn.Conv1d(

                self.en_residual_channel,
                self.en_dilation_channel,
                self.filter_width, 
                dilation = dilation,
                bias=self.use_bias
                ))

            self.en_dense_layer_stack.append( nn.Conv1d(

                self.en_dilation_channel, 
                self.en_residual_channel, 
                1,
                bias = self.use_bias
                ))


    def forward(self,sample):

        sample = self.en_causal_layer(sample)

        for i,(dilation_layer,dense_layer) in enumerate(zip(self.en_dilation_layer_stack,self.en_dense_layer_stack)):

            current = sample
            sample = F.relu(sample)
            sample = dilation_layer(sample)
            sample = F.relu(sample)
            sample = dense_layer(sample)
            _,_,current_length = sample.size()
            current_in_sliced = current[:,:,-current_length:]
            sample = sample + current_in_sliced

        sample = self.bottleneck_layer(sample)  
        sample = self.pool1d(sample)
        return sample
    
class Decoder(nn.Module):
    
    def __init__(self, 

        filter_width,
        quantization_channel,
        dilations,
                 
        en_bottleneck_width,

        de_residual_channel,
        de_dilation_channel,
        de_skip_channel,

        use_bias,
        cuda_available,
        receptive_field):

        super(Decoder, self).__init__()

        self.filter_width = filter_width
        self.quantization_channel = quantization_channel
        self.dilations = dilations

        self.en_bottleneck_width = en_bottleneck_width
        self.de_residual_channel = de_residual_channel
        self.de_dilation_channel = de_dilation_channel
        self.de_skip_channel = de_skip_channel

        self.use_bias = use_bias
        self.cuda_available = cuda_available
        self.receptive_field = receptive_field
        
        self.softmax = nn.Softmax(dim=1)
        
        self.de_dilation_layer_stack = nn.ModuleList()

        for i,dilation in enumerate(self.dilations):

            current={}

            current['filter_gate'] = nn.Conv1d(
                self.de_residual_channel,
                2*self.de_dilation_channel, 
                self.filter_width,
                dilation =dilation,
                bias = self.use_bias)

            current['dense'] =nn.Conv1d(
                self.de_dilation_channel, 
                self.de_residual_channel, 
                kernel_size =1,
                dilation =dilation,
                bias=self.use_bias)

            current['skip'] = nn.Conv1d(
                self.de_dilation_channel, 
                self.de_skip_channel,
                dilation = dilation,
                kernel_size=1,
                bias = self.use_bias)

            self.de_dilation_layer_stack.extend(list(current.values()))
            
        self.connection_1 = nn.Conv1d(self.de_skip_channel,self.de_skip_channel,1,bias=self.use_bias)
        self.connection_2 = nn.Conv1d(self.de_skip_channel,self.quantization_channel,1,bias=self.use_bias)
        self.de_causal_layer = nn.Conv1d(self.quantization_channel,self.de_residual_channel,self.filter_width,bias = self.use_bias)

        if self.cuda_available:
            self.conv1d =  nn.Conv1d(self.en_bottleneck_width, 2*self.de_dilation_channel,1).cuda()
            self.conv2d =  nn.Conv1d(self.en_bottleneck_width, self.de_skip_channel,1).cuda()
        else:
            self.conv1d =  nn.Conv1d(self.en_bottleneck_width, 2*self.de_dilation_channel,1)
            self.conv2d =  nn.Conv1d(self.en_bottleneck_width, self.de_skip_channel,1)
        
        
    def _conditon(self,x,encoding):

        mb,channels,length = encoding.size()
        xlength=x.size()[2]

        if xlength %length == 0:
            encoding = encoding.view(mb,channels,length,1)
		
            x = x.view(mb,channels,length,-1)
            x = x + encoding
            x = x.view(mb,channels,xlength)
            del encoding
            
        else:
            repeat_num = int(np.floor(xlength/length))
            encoding_repeat = encoding.repeat(1,1,repeat_num)
            encoding_repeat = torch.cat((encoding_repeat,encoding[:,:,:xlength%length]),2)
            x = x + encoding_repeat
            del encoding_repeat
        return x
    
    def forward(self,sample,encoding,output_width):

        current_out = self.de_causal_layer(sample)
        skip_contribute_stack = []

        for i, dilation in enumerate(self.dilations):

            j = 3*i
            current_in  = current_out

            filter_gate_layer,  dense_layer, skip_layer = \
                self.de_dilation_layer_stack[j], \
                self.de_dilation_layer_stack[j+1], \
                self.de_dilation_layer_stack[j+2]

            sample = filter_gate_layer(current_in)
            en = self.conv1d(encoding)

            sample = self._conditon(sample,en)
            _,channels,_ = sample.size()

            xg = sample[:,:-int(channels/2),:]
            xf = sample[:,-int(channels/2):,:]
            z = F.tanh(xf)*F.sigmoid(xg)

            x_res = dense_layer(z)

            _,_,slice_length = x_res.size()

            current_in_sliced = current_in[:,:,-slice_length:]
            current_out = current_in_sliced + x_res

            skip = z[:,:,-output_width:]

            skip = skip_layer(skip)

            skip_contribute_stack.append(skip)

        result = sum(skip_contribute_stack)
        result=F.relu(result)
        result = self.connection_1(result)
        
        en = self.conv2d(encoding)

        result = self._conditon(result,en)
        result= F.relu(result)
        result = self.connection_2(result)
        batch_size, channels, seq_len = result.size()
        result = result.view(-1, self.quantization_channel)
        result = self.softmax(result)
        
        return result
