import torch
import torch.nn as nn

from .Unet_3Dblock import Encoder, Decoder, ConnectBridge, Bridge, ROIDecoder, MaskDecoder


class TraditionUnet(nn.Module):
    r'''
    Traditional 3D Unet
    Return the encoded features
    Args:
        num_layers: the channel of each layer
        dim_input: the input dim
        dim_output: the output dim
        kernel_size: the size of convolution kernel
        dropout: the dropout ratio
    '''
    def __init__(self, num_layers:list, dim_input:int, dim_output:int, 
                 kernel_size:int=3, dropout:float=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.encode = Encoder(num_layers=self.num_layers, dim_input=self.dim_input, kernel_size=self.kernel_size,
                              dropout=self.dropout)
        self.decode = Decoder(num_layers=self.num_layers, dim_output=self.dim_output, kernel_size=self.kernel_size, 
                              dropout=self.dropout)
    
    def forward(self, x):
        bottle_block, inter_block = self.encode(x)
        out = self.decode(bottle_block, inter_block)
        return out


class BottleTransUnet(nn.Module):
    r'''
    Trans Unet of bottle
    Return the encoded features
    Args:
        num_layers: the channel of each layer
        dim_input: the input dim
        dim_output: the output dim
        kernel_size: the size of convolution kernel
        dropout: the dropout ratio
    '''
    def __init__(self, num_layers:list, dim_input:int, dim_output:int, 
                 kernel_size:int=3, dropout:float=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.encode = Encoder(num_layers=self.num_layers, dim_input=self.dim_input, kernel_size=self.kernel_size,
                              dropout=self.dropout)
        self.decode = Decoder(num_layers=self.num_layers, dim_output=self.dim_output, kernel_size=self.kernel_size, 
                              dropout=self.dropout)
        self.connect_bridge = ConnectBridge(d_model=self.num_layers[-1], nhead=8, dropout=self.dropout, N=4)
    
    def forward(self, x):
        bottle_block, inter_block = self.encode(x)
        connect_bottle = self.connect_bridge(bottle_block)
        out = self.decode(connect_bottle, inter_block)
        return out


class SkipTransUnet(nn.Module):
    r'''
    Trans Unet of bottle
    Return the encoded features
    Args:
        num_layers: the channel of each layer
        dim_input: the input dim
        dim_output: the output dim
        kernel_size: the size of convolution kernel
        dropout: the dropout ratio
    '''
    def __init__(self, num_layers:list, dim_input:int, dim_output:int, 
                 kernel_size:int=3, dropout:float=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.encode = Encoder(num_layers=self.num_layers, dim_input=self.dim_input, kernel_size=self.kernel_size,
                              dropout=self.dropout)
        self.decode = Decoder(num_layers=self.num_layers, dim_output=self.dim_output, kernel_size=self.kernel_size, 
                              dropout=self.dropout)
        self.connect_bridge_list = nn.ModuleList([
                                        ConnectBridge(d_model=self.num_layers[i], 
                                                      nhead=8, 
                                                      dropout=self.dropout, 
                                                      N=4)
                                            for i in range(len(self.num_layers))])
    
    def forward(self, x):
        bottle_block, inter_block = self.encode(x)
        inter_block_list = []
        for i in range(len(self.num_layers)):
            if i != (len(self.num_layers) -1):
                inter_block_list.append(self.connect_bridge_list[i](inter_block[i]))
            else:
                connect_bottle = self.connect_bridge_list[i](bottle_block)
        out = self.decode(connect_bottle, inter_block)
        return out


class MaskSkipTransUnet(nn.Module):
    r'''
    Trans Unet of skip connection with mask
    Return the encoded features
    Args:
        num_layers: the channel of each layer
        dim_input: the input dim
        dim_output: the output dim
        kernel_size: the size of convolution kernel
        dropout: the dropout ratio
    '''
    def __init__(self, num_layers:list, dim_input:int, dim_output:int, 
                 kernel_size:int=3, dropout:float=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.encode = Encoder(num_layers=self.num_layers, dim_input=self.dim_input, kernel_size=self.kernel_size,
                              dropout=self.dropout)
        self.decode = Decoder(num_layers=self.num_layers, dim_output=self.dim_output, kernel_size=self.kernel_size, 
                              dropout=self.dropout)
        self.connect_bridge = Bridge(num_layers=self.num_layers,
                                     nhead_lens=16)

    def forward(self, x):
        bottle_block, inter_block = self.encode(x)
        connect_bottle, inter_block, mask_list = self.connect_bridge(bottle_block, inter_block)
        out = self.decode(connect_bottle, inter_block)
        # add loss for mask list in the future
        return out, mask_list
        # return out


class MaskTransUnet(nn.Module):
    r'''
    Trans Unet of skip connection with mask using upsamling
    Return the encoded features
    Args:
        num_layers: the channel of each layer
        dim_input: the input dim
        dim_output: the output dim
        kernel_size: the size of convolution kernel
        dropout: the dropout ratio
    '''
    def __init__(self, num_layers:list, roi_size_list:list, is_roi_list:list, dim_input:int, dim_output:int, 
                 kernel_size:int=3, dropout:float=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.roi_size_list = roi_size_list
        self.is_roi_list = is_roi_list

        self.encode = Encoder(num_layers=self.num_layers, dim_input=self.dim_input, kernel_size=self.kernel_size,
                              dropout=self.dropout)
        self.decode = ROIDecoder(num_layers=self.num_layers,
                                 roi_size_list=self.roi_size_list,
                                 is_roi_list=self.is_roi_list,
                                 dim_output=self.dim_output,
                                 dropout=self.dropout)
        # self.mask_decoder = MaskDecoder(num_layers=num_layers)

    def forward(self, x):
        bottle_block, inter_block = self.encode(x)
        out, mask_list = self.decode(bottle_block, inter_block)
        '''
        mask_list = self.mask_decoder(mask_list)
        # add loss for mask list in the future
        # return out, mask_list
        # return out
        
        out_binary = (out>=0.5).float()
        
        uncertainty_list = [(mask_list[i]-out_binary)**2 for i in range(len(mask_list))]
        uncertainty_list.append((out-out_binary)**2)
        uncertainty = sum(uncertainty_list) / len(uncertainty_list)
        '''
        if self.training:
            return out, mask_list
        else:
            max_idx = torch.argmax(out, dim=1, keepdim=True)
            out = 0*out
            out.scatter_(1, max_idx, 1)
            return out
            # return mask_list[-1]
            # return uncertainty


Model_Dict = {
    'TraditionUnet': TraditionUnet,
    'BottleTransUnet': BottleTransUnet,
    'SkipTransUnet': SkipTransUnet,
    'MaskSkipTransUnet': MaskSkipTransUnet,
    'MaskTransUnet':MaskTransUnet
}

def get_model_dict(name: str):
    '''
    Return the loss dict from name list
    Args:
        name_list: name list
    '''
    model_fn = Model_Dict[name]
    return model_fn