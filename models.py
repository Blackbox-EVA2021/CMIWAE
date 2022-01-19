#! /usr/bin/env python

# Models

import torch
from torch import nn, Tensor
from typing import Type, List

class IllegalArgument(Exception):
    pass

# Encoder and decoder parts for variational autoencoder

def conv_BN(in_ch, out_ch, kernel_size, stride, use_BatchNorm):
    output = [nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)]
    if use_BatchNorm:
        output.append(nn.BatchNorm2d(out_ch))
    return output

class MinimalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        nonlin = nn.ReLU(inplace=True),
        use_BatchNorm: bool = True
    ) -> None:
        super(MinimalBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        assert (in_ch >= 1) and (out_ch >= 1) and (kernel_size >= 1) and (stride >= 1)
        
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)]
        layers.append(nonlin)
        if use_BatchNorm:
            layers.append(nn.BatchNorm2d(out_ch))
        self.conv_nonlin_bn = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_nonlin_bn(x)
    
    
def deconv_BN(in_ch, out_ch, kernel_size, stride, use_BatchNorm=True):
    if stride == 1:
        output_padding = 0
    elif stride == 2:
        output_padding = kernel_size % 2
    else:
        raise IllegalArgument(f"Unexpected value {stride} for stride.")
        
    output = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding=(kernel_size - 1) // 2, output_padding=output_padding)]
    if use_BatchNorm:
        output.append(nn.BatchNorm2d(out_ch))
    return output
    
class MinimalBlockTranspose(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        nonlin = nn.ReLU(inplace=True),
        use_BatchNorm: bool = True
    ) -> None:
        super(MinimalBlockTranspose, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        assert (in_ch >= 1) and (out_ch >= 1) and (kernel_size >= 1) and (stride >= 1)
              
        if stride == 1:
            output_padding = 0
        elif stride == 2:
            output_padding = kernel_size % 2
        else:
            raise IllegalArgument(f"Unexpected value {stride} for stride.")
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding=(kernel_size - 1) // 2, output_padding=output_padding)]
        layers.append(nonlin)
        if use_BatchNorm:
            layers.append(nn.BatchNorm2d(out_ch))
        self.deconv_nonlin_bn = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv_nonlin_bn(x)

    
    
class Enc(nn.Module):  # plain encoder, without additional level inputs or outputs

    def __init__(
        self,
        in_ch: int,
        block: Type[MinimalBlock],
        height: int = 64,
        width: int = 128,
        layers_num: List[int] = [1, 1, 1, 1, 1, 1, 1],
        channels: List[int] = [32, 48, 64, 96, 128, 192, 256],
        latent_ch: int = 2,
        latent_size: int = 64,
        kernel_size: int = 3,
        nonlin = nn.ReLU(inplace=True),
        use_BatchNorm: bool = True,
        dropout: float = 0.1,
        fc_hidden_layer: int = None,
        init_params = False
    ) -> None:
        super(Enc, self).__init__()
        
        assert (height >= 1) and (width >=1)
        self.length = len(layers_num)  # number of outputs, number of reductions (except first), number of block clusters
        assert (self.length == len(channels)) and (self.length >= 1)
        assert (in_ch >= 1) and (latent_ch >= 1) and (latent_size >= 1) and (kernel_size >= 1)

        reduction_factor = 2**(self.length - 1)
        assert (height % reduction_factor == 0) and (width % reduction_factor == 0)
        
        self.dropout_lay = torch.nn.Dropout(dropout)
        
        self.layer = []
        input_ch = in_ch
        for i in range(self.length):
            self.layer.append(self._make_layer(block, layers_num[i], input_ch, channels[i], kernel_size, 1 if i == 0 else 2, nonlin, use_BatchNorm))
            input_ch = channels[i]
        self.layer = nn.ModuleList(self.layer)
        
        self.fc_input_size = channels[-1] * (height // reduction_factor) * (width // reduction_factor)
        if fc_hidden_layer is None:
            self.fc = nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(self.fc_input_size, latent_ch * latent_size),
                torch.nn.Unflatten(1, (latent_ch, latent_size)))
        else:
            assert fc_hidden_layer >= 1
            self.fc = nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(self.fc_input_size, fc_hidden_layer),
                torch.nn.BatchNorm1d(fc_hidden_layer),
                nonlin,
                torch.nn.Linear(fc_hidden_layer, latent_ch * latent_size),
                torch.nn.Unflatten(1, (latent_ch, latent_size)))            

        if init_params:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: Type[MinimalBlock], blocks: int, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                    nonlin, use_BatchNorm: bool)-> nn.Sequential:
        assert blocks >= 1
        
        layers = []
        layers.append(block(in_ch, out_ch, kernel_size, stride, nonlin, use_BatchNorm))
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, kernel_size, 1, nonlin, use_BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.dropout_lay(x)
        for i in range(self.length):
            out = self.layer[i](out)
        out = self.fc(out)
        return out
    
    def count_num_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_fc_input_size(self):
        return self.fc_input_size

    
class Enc_cond(nn.Module):  # encoder for conditional data

    def __init__(
        self,
        in_ch: int,
        block: Type[MinimalBlock],
        height: int = 64,
        width: int = 128,
        layers_num: List[int] = [1, 1, 1, 1, 1, 1, 1],
        channels: List[int] = [32, 48, 64, 96, 128, 192, 256],
        latent_ch: int = 2,
        latent_size: int = 64,
        kernel_size: int = 3,
        nonlin = nn.ReLU(inplace=True),
        use_BatchNorm: bool = True,
        dropout: float = 0.1,
        init_params = False
    ) -> None:
        super(Enc_cond, self).__init__()
        
        assert (height >= 1) and (width >=1)
        self.length = len(layers_num)  # number of outputs, number of reductions (except first), number of block clusters
        assert (self.length == len(channels)) and (self.length >= 1)
        assert (in_ch >= 1) and (latent_ch >= 1) and (latent_size >= 1) and (kernel_size >= 1)

        reduction_factor = 2**(self.length - 1)
        assert (height % reduction_factor == 0) and (width % reduction_factor == 0)
        
        self.dropout_lay = torch.nn.Dropout(dropout)
        
        self.layer = []
        input_ch = in_ch
        for i in range(self.length):
            self.layer.append(self._make_layer(block, layers_num[i], input_ch, channels[i], kernel_size, 1 if i == 0 else 2, nonlin, use_BatchNorm))
            input_ch = channels[i]
        self.layer = nn.ModuleList(self.layer)
        
        self.fc_input_size = channels[-1] * (height // reduction_factor) * (width // reduction_factor)
        self.fc_params = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.fc_input_size, latent_ch * latent_size),
            torch.nn.Unflatten(1, (latent_ch, latent_size))
        )

        if init_params:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: Type[MinimalBlock], blocks: int, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                    nonlin, use_BatchNorm: bool)-> nn.Sequential:
        assert blocks >= 1
        
        layers = []
        layers.append(block(in_ch, out_ch, kernel_size, stride, nonlin, use_BatchNorm))
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, kernel_size, 1, nonlin, use_BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        output = [x]
        out = self.dropout_lay(x)
        for i in range(self.length):
            out = self.layer[i](out)
            output.append(out)
        assert len(output) == self.length + 1
        params = self.fc_params(out)
        return output, params
    
    def count_num_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
class Dec(nn.Module):  # plain decoder, without additional level inputs

    def __init__(
        self,
        out_ch: int,
        block: Type[MinimalBlockTranspose],
        height: int = 64,
        width: int = 128,
        layers_num: List[int] = [1, 1, 1, 1, 1, 1, 1],
        channels: List[int] = [32, 48, 64, 96, 128, 192, 256],
        latent_size: int = 64,
        kernel_size: int = 3,
        nonlin = nn.ReLU(inplace=True),
        use_BatchNorm: bool = True,
        dropout: float = 0.1,
        init_params = False
    ) -> None:
        super(Dec, self).__init__()
        
        assert (height >= 1) and (width >=1)
        self.length = len(layers_num)  # number of outputs, number of reductions (except first), number of block clusters
        assert (self.length == len(channels)) and (self.length >= 1)
        assert (out_ch >= 1) and (latent_size >= 1) and (kernel_size >= 1)

        reduction_factor = 2**(self.length - 1)
        assert (height % reduction_factor == 0) and (width % reduction_factor == 0)
        
        self.dropout_lay = torch.nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            torch.nn.Linear(latent_size, channels[-1] * (height // reduction_factor) * (width // reduction_factor)),
            nonlin,
            torch.nn.Unflatten(1, (channels[-1], (height // reduction_factor), (width // reduction_factor)))
        )
        
        self.layer = []
        input_ch = channels[-1]
        for i in range(self.length - 1, 0, -1):
            self.layer.append(self._make_layer(block, layers_num[i], channels[i], channels[i - 1], kernel_size, 2, nonlin, use_BatchNorm))
        self.layer.append(self._make_layer(block, layers_num[i], channels[0], out_ch, kernel_size, 1, nonlin, use_BatchNorm))
        self.layer = nn.ModuleList(self.layer)
        
        if init_params:
            for m in self.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: Type[MinimalBlockTranspose], blocks: int, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                    nonlin, use_BatchNorm: bool)-> nn.Sequential:
        assert blocks >= 1
        
        layers = []
        layers.append(block(in_ch, out_ch, kernel_size, stride, nonlin, use_BatchNorm))
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, kernel_size, 1, nonlin, use_BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.dropout_lay(x)
        out = self.fc(out)
        for i in range(self.length):
            out = self.layer[i](out)
        return out
    
    def count_num_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
class Dec_cond(nn.Module):  # conditional decoder, with additional level inputs of conditional data from conditional encoder

    def __init__(
        self,
        out_ch: int,
        block: Type[MinimalBlockTranspose],
        height: int = 64,
        width: int = 128,
        layers_num: List[int] = [1, 1, 1, 1, 1, 1, 1],
        channels: List[int] = [32, 48, 64, 96, 128, 192, 256],
        cond_in_ch: List[int] = [32, 48, 64, 96, 128, 192, 256],
        latent_size: int = 64,
        kernel_size: int = 3,
        nonlin = nn.ReLU(inplace=True),
        use_BatchNorm: bool = True,
        dropout: float = 0.1,
        init_params = False
    ) -> None:
        super(Dec_cond, self).__init__()
        
        assert (height >= 1) and (width >=1)
        self.length = len(layers_num)  # number of outputs, number of reductions (except first), number of block clusters
        assert (self.length == len(channels)) and (self.length >= 1)
        assert (out_ch >= 1) and (latent_size >= 1) and (kernel_size >= 1)
        
        self.cond_in_ch = cond_in_ch

        reduction_factor = 2**(self.length - 1)
        assert (height % reduction_factor == 0) and (width % reduction_factor == 0)
        
        self.dropout_lay = torch.nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            torch.nn.Linear(latent_size, channels[-1] * (height // reduction_factor) * (width // reduction_factor)),
            nonlin,
            torch.nn.Unflatten(1, (channels[-1], (height // reduction_factor), (width // reduction_factor)))
        )
        
        self.layer = []
        for i in range(self.length - 1, 0, -1):
            self.layer.append(self._make_layer(block, layers_num[i], channels[i] + cond_in_ch[i], channels[i - 1], kernel_size, 2, nonlin, use_BatchNorm))
        self.layer.append(self._make_layer(block, layers_num[i], channels[0] + cond_in_ch[0], out_ch, kernel_size, 1, nonlin, use_BatchNorm))
        self.layer = nn.ModuleList(self.layer)
        
        if init_params:
            for m in self.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: Type[MinimalBlockTranspose], blocks: int, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                    nonlin, use_BatchNorm: bool)-> nn.Sequential:
        assert blocks >= 1
        
        layers = []
        layers.append(block(in_ch, out_ch, kernel_size, stride, nonlin, use_BatchNorm))
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, kernel_size, 1, nonlin, use_BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, xl: List[Tensor]) -> Tensor:
        assert len(xl) == self.length + 1
        out = self.dropout_lay(x)
        out = self.fc(out)
        for i in range(self.length):
            out = torch.cat([out, xl[self.length - i][:, :self.cond_in_ch[(self.length - 1) - i]]], 1)
            out = self.layer[i](out)
        return out
    
    def count_num_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 