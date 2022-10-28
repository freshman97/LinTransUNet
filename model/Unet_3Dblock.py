import torch
from torch.functional import Tensor
import torch.nn as nn

import torch.nn.functional as F
from .trans_block import SelfAttentionLayer, Conv3dPosEmbedding, TransEncoder, clones
from typing import Optional, Union, Tuple


def get_min_max_indice2(mask_one_1d:Tensor, threshold:float=1e-5):
    frame_ratio = mask_one_1d / torch.max(mask_one_1d)
    frame_index = torch.where((frame_ratio <= threshold))[0]
    max_indice = torch.argmax(mask_one_1d)
    if frame_index.numel() == 0:
        return 0*max_indice, frame_ratio.size(0)-1, max_indice

    '''
    Sometimes life sucks
    print(frame_ratio)
    print(frame_index)
    print(max_indice)
    '''
    left = frame_index <= max_indice
    if torch.sum(left) != 0:
        min_index = torch.max(frame_index[left])
    else:
        min_index = 0*max_indice

    right = frame_index >= max_indice
    if torch.sum(right) != 0:
        max_index = torch.min(frame_index[right])
    else:
        max_index = frame_ratio.size(0) - 1
    
    return min_index, max_index, max_indice

def get_min_max_indice(mask_one_1d:Tensor, threshold:float=0.001):
    if torch.sum(mask_one_1d) == 0:
        # print(mask_one_1d.shape)
        mid_value = torch.tensor(mask_one_1d.shape[0]/2, device=mask_one_1d.device)
        return mid_value-1, mid_value+1, mid_value

    frame_ratio = torch.cumsum(mask_one_1d, dim=0) / torch.sum(mask_one_1d)
    min_index = torch.searchsorted(frame_ratio, threshold, right=False)
    max_index = torch.searchsorted(frame_ratio, 1-threshold, right=True)

    max_indice = torch.searchsorted(frame_ratio, 0.5, right=True)

    return min_index, max_index, max_indice

def get_transfer_index(x0, x1, h, roi_size, eval_roi_size, device):
    img_index = torch.arange(0, eval_roi_size, device=device, dtype=torch.float32)

    k2 = (x1 - x0) / (roi_size-1)
    k1 = (h - x1 + x0) / (eval_roi_size - roi_size)
    out_index = img_index * k2 + x0*(1-k2/k1)
    indice_bool = out_index <= x0
    out_index2 = out_index*(k1/k2) + x0*(1-k1/k2)
    out_index[indice_bool] = out_index2[indice_bool]
    indice_bool = out_index >= x1
    out_index2 = out_index*(k1/k2) + x1*(1-k1/k2)
    out_index[indice_bool] = out_index2[indice_bool]
    out_index = out_index*2./h - 1
    return out_index

def get_transfer_back_index(x0, x1, h, roi_size, eval_roi_size, device):
    img_index = torch.arange(0, h+1, device=device, dtype=torch.float32)

    k2 = roi_size / (x1 - x0)
    k1 = (eval_roi_size - roi_size) / (h - x1 + x0)
    p0 = x0*k1
    p1 = eval_roi_size - (h-x1)*k1
    out_index = img_index * k2 + p0*(1-k2/k1)
    indice_bool = out_index <= p0
    out_index2 = out_index*(k1/k2) + p0*(1-k1/k2)
    out_index[indice_bool] = out_index2[indice_bool]
    indice_bool = out_index >= p1
    out_index2 = out_index*(k1/k2) + p1*(1-k1/k2)
    out_index[indice_bool] = out_index2[indice_bool]

    out_index = out_index*2/eval_roi_size - 1
    return out_index

def get_solid_transfer_index(x0, x1, h, eval_roi_size, device):
    img_index = torch.arange(0, eval_roi_size, device=device, dtype=torch.float32, requires_grad=False)

    k2 = 1
    k1 = (h - x1 + x0) / (eval_roi_size-1 - x1 + x0)
    out_index = img_index * k2 + x0*(1-k2/k1)
    indice_bool = out_index <= x0
    out_index2 = img_index * k1
    out_index[indice_bool] = out_index2[indice_bool]
    indice_bool = out_index >= x1
    out_index2 = (img_index - eval_roi_size+1) * k1 + h 
    out_index[indice_bool] = out_index2[indice_bool]
    out_index = out_index*2/h - 1
    '''
    if torch.any(torch.isnan(out_index)):
        print('roi transfer')
        print(x0, x1, h)
    '''
    return out_index

def get_solid_back_index(x0, x1, h, eval_roi_size, device):
    img_index = torch.arange(0, h+1, device=device, dtype=torch.float32, requires_grad=False)

    k2 = 1
    k1 = (eval_roi_size-1 - x1 + x0) / (h - x1 + x0)
    p0 = x0*k1
    p1 = eval_roi_size-1 - (h-x1)*k1
    out_index = img_index * k2 + p0*(1-k2/k1)
    indice_bool = out_index <= p0
    out_index2 = out_index*(k1/k2) + p0*(1-k1/k2)
    out_index[indice_bool] = out_index2[indice_bool]
    indice_bool = out_index >= p1
    out_index2 = out_index*(k1/k2) + p1*(1-k1/k2)
    out_index[indice_bool] = out_index2[indice_bool]
    out_index = out_index*2/(eval_roi_size-1) - 1

    return out_index


def windows_embedding(img:Tensor, kernel_size:int=2):
    '''
    Embedding the image in the ratio of kernel size
    Input
        Image: size [nbatch, 1, h, w, d]
        kernel_size: the embedding window size default 2
    Output:
        Image: size [nbatch, kernel_size**2, h//kernel_size, w//kernel_size, d]
    '''
    nbatch, _, h, w, d = img.shape
    img = torch.reshape(img, (nbatch, h//kernel_size, kernel_size, w//kernel_size, kernel_size, d))
    img = img.permute((0, 2, 4, 1, 3, 5))
    img = img.flatten(start_dim=1, end_dim=2)
    return img

def windows_unembedding(img:Tensor, kernel_size:int=2):
    '''
    Unembedding the image in the ratio of kernel size
    Input
        Image: size [nbatch, channel, h//kernel_size, w//kernel_size, d]
        kernel_size: the embedding window size default 2
    Output:
        Image: size [nbatch, channel//kernel_size**2, h, w, d]
    '''
    nbatch, channel, h, w, d = img.shape
    img = torch.reshape(img, (nbatch, channel//kernel_size**2, kernel_size, kernel_size, h,  w, d))
    img = img.permute((0, 1, 4, 2, 5, 3, 6))
    img = img.flatten(start_dim=2, end_dim=3)
    img = img.flatten(start_dim=3, end_dim=4)
    return img

class Attention3DBlock(nn.Module):
    def __init__(self, in_dim, d_model, nhead:int, dropout:float=0.3, N:int=8):
        '''
        Here is used to define the connection of skip encoded feature maps
        The transformer block will be only applied on the selected roi region
        return should be the same size with input
        Args:
            in_dim: input dimension from convolution
            d_model: the dimension for the model, use the bottle layer
            nhead: head for multihead attention
            N: attention repeated times
        Output:
            return the reshaped 3d convolutional block
        '''
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.pos_encode = Conv3dPosEmbedding(dim=d_model, dropout=dropout, emb_kernel=3)
        attn_layer = SelfAttentionLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout)
        self.transformer = TransEncoder(attn_layer, N)

    def forward(self, x: torch.Tensor, mask:Optional[torch.Tensor]=None):
        '''
        x: Input x size
            [nbatch, channel, height, width, depth]
        Mask:
            [nbatch, 1, height, width, depth]
        '''
        nbatch, _, height, width, depth = x.shape

        x = self.pos_encode(x)

        x = x.flatten(start_dim=2).transpose_(1, 2)
        if mask is not None:
            mask = mask.flatten(start_dim=2).transpose_(1, 2)
        x = self.transformer(x, mask=mask)
        x = x.transpose_(1, 2).reshape(nbatch, -1, height, width, depth)
        return x


class SpatialAttention3DBlock(nn.Module):
    def __init__(self, in_channel1, in_channel2, 
                 inter_channel, dim_output, kernel_size:int=1, 
                 stride:int=1):
        super().__init__()
        self.W_x = nn.Sequential(
                                 nn.Conv3d(in_channels=in_channel1, 
                                           out_channels=inter_channel,
                                           kernel_size=kernel_size,
                                           stride=stride),
                                 nn.InstanceNorm3d(inter_channel))
        self.W_g = nn.Sequential(
                                 nn.Conv3d(in_channels=in_channel2, 
                                           out_channels=inter_channel,
                                           kernel_size=kernel_size,
                                           stride=stride),
                                 nn.InstanceNorm3d(inter_channel))
        self.psi = nn.Sequential(nn.Conv3d(in_channels=inter_channel, 
                                           out_channels=1,
                                           kernel_size=kernel_size,
                                           stride=stride),
                                 nn.Sigmoid())

    def forward(self, x, up):
        x = self.W_x(x)
        up = self.W_g(up)
        x = self.psi(F.relu(x+up, inplace=True))
        return x


class PosAttention3DBlock(nn.Module):
    def __init__(self, in_dim, d_model, nhead:int, dropout:float=0.3, N:int=8):
        '''
        Here is used to define the connection of skip encoded feature maps
        The transformer block will be only applied on the selected roi region
        return should be the same size with input
        Args:
            in_dim: input dimension from convolution
            d_model: the dimension for the model, use the bottle layer
            nhead: head for multihead attention
            N: attention repeated times
        Output:
            return the reshaped 3d convolutional block
        '''
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.N = N
        # self.pos_encoder = Conv3dPosEmbedding(dim=d_model, dropout=dropout, emb_kernel=3)
        pos_encoder = Conv3dPosEmbedding(dim=d_model, dropout=dropout, emb_kernel=3)
        self.pos_encoders = clones(pos_encoder, N)
        attn_layer = SelfAttentionLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout)
        self.layers = clones(attn_layer, N)
        # self.pos_encodes = clones(pos_encode, N)

    def forward(self, x: torch.Tensor, mask:Optional[torch.Tensor]=None):
        '''
        x: Input x size
            [nbatch, channel, height, width, depth]
        Mask:
            [nbatch, 1, height, width, depth]
        '''
        nbatch, _, height, width, depth = x.shape

        # reshape the position to depth, height, width
        x = x.permute((0, 1, 4, 2, 3))
        if mask is not None:
            mask = mask.permute((0, 1, 4, 2, 3))
            mask = mask.flatten(start_dim=2).transpose_(1, 2)

        x = x.flatten(start_dim=2).transpose_(1, 2)
        for i in range(self.N):
            x = self.layers[i](x, mask)
            if i == 0:
                x = x.transpose_(1, 2).reshape(nbatch, -1, depth, height, width)
                x = self.pos_encoders[i](x)
                x = x.flatten(start_dim=2).transpose_(1, 2)

        x = x.transpose_(1, 2).reshape(nbatch, -1, depth, height, width)
        x = x.permute((0, 1, 3, 4, 2))
        return x


class SolidBlock(nn.Module):
    def __init__(self, num_layer, inter_num:int=12):
        super().__init__()
        self.solid_layer = nn.Sequential(*[
                                           nn.Linear(in_features=num_layer, out_features=inter_num),
                                           nn.BatchNorm1d(num_features=inter_num),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(in_features=inter_num, out_features=1),])

    def forward(self, x):
        x = torch.sigmoid(self.solid_layer(x))
        return x

class DownBlock(nn.Module):
    '''
    Downsampling block for UNet
    Args:
        in_channels: input channel of 3D convolution block
        out_channels: output channel of 3D convolution block
        kernel_size: the kernel size for convolution
        stride: stride for convolution
        paddding: the padding for convolution
        dropout: dropout ratio
        is_res: whether apply res block or not
        activation: apply relu or leakyrelu activation function
        dropout
    '''
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int = 3, stride:Union[int, Tuple[int,...]]=2,
                 padding: int = 0, dropout:float=None, is_res:bool = True, 
                 activation='leakyrelu'):
        super().__init__()
        self.is_res = is_res

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                             stride=1, padding='same')
        self.norm1 = nn.InstanceNorm3d(num_features=in_channels)

        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        x_conv1 = self.norm1(self.conv1(x))
        x_conv1 = self.activation(x_conv1)

        if self.is_res:
            x_conv1 = x_conv1 + x
        else:
            x_conv1 = x_conv1

        x = self.norm2(self.conv2(x_conv1))
        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x, x_conv1

class DownEmbedBlock(nn.Module):
    '''
    DownEmbedsampling block for UNet
    Args:
        in_channels: input channel of 3D convolution block
        out_channels: output channel of 3D convolution block
        down_times: the downsampling scale
        kernel_size: the kernel size for convolution
        stride: stride for convolution
        paddding: the padding for convolution
        dropout: dropout ratio
        is_res: whether apply res block or not
        activation: apply relu or leakyrelu activation function
        dropout
    '''
    def __init__(self, in_channels:int, out_channels:int, down_times:int, kernel_size:int = 3, stride:Union[int, Tuple[int,...]]=1,
                 padding: int = 0, dropout:float=None, activation='leakyrelu'):
        super().__init__()
        self.down_times = down_times
        self.channel_list = [in_channels*(2**i) if in_channels*(2**i)<=out_channels else out_channels for i in range(self.down_times+1)]
        self.channel_list[-1] = out_channels

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        self.module_list = nn.Sequential(*[
                                        nn.Sequential(
                                            nn.Conv3d(in_channels=self.channel_list[i],
                                                      out_channels=self.channel_list[i+1],
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding),
                                            nn.InstanceNorm3d(num_features=self.channel_list[i+1]),
                                            self.activation,
                                            self.dropout) for i in range(down_times)])

    def forward(self, x):
        return self.module_list(x)


class UpEmbedBlock(nn.Module):
    '''
    UpEmbedsampling block for UNet
    Args:
        in_channels: input channel of 3D convolution block
        out_channels: output channel of 3D convolution block
        down_times: the downsampling scale
        kernel_size: the kernel size for convolution
        stride: stride for convolution
        paddding: the padding for convolution
        dropout: dropout ratio
        activation: apply relu or leakyrelu activation function
        dropout
    '''
    def __init__(self, in_channels:int, out_channels:int, down_times:int, kernel_size:int = 3, stride:Union[int, Tuple[int,...]]=1,
                 padding: int = 0, dropout:float=None, is_res:bool = True, 
                 activation='leakyrelu'):
        super().__init__()
        self.is_res = is_res
        self.down_times = down_times
        self.channel_list = [out_channels//(2**i) if out_channels//(2**i)>=in_channels else out_channels for i in range(self.down_times+1)]
        self.channel_list[-1] = in_channels

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        self.module_list = nn.Sequential(*[
                                        nn.Sequential(
                                            nn.Upsample(scale_factor=stride),
                                            nn.Conv3d(in_channels=self.channel_list[i],
                                                      out_channels=self.channel_list[i+1],
                                                      kernel_size=kernel_size,
                                                      stride=1,
                                                      padding=padding),
                                            nn.InstanceNorm3d(num_features=self.channel_list[i+1]),
                                            self.activation,
                                            self.dropout) for i in range(down_times)])

    def forward(self, x):
        return self.module_list(x)


class EmbedAttention3DBlock(nn.Module):
    def __init__(self, in_dim, d_model, nhead:int, dropout:float=0.3, N:int=8):
        '''
        Here is used to define the connection of skip encoded feature maps
        The transformer block will be only applied on the selected roi region
        return should be the same size with input
        Args:
            in_dim: input dimension from convolution
            d_model: the dimension for the model, use the bottle layer
            nhead: head for multihead attention
            N: attention repeated times
        Output:
            return the reshaped 3d convolutional block
        '''
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        # self.down_times = int(math.log2(d_model // in_dim))
        # assert d_model == in_dim*(2**self.down_times), 'the embed dimension should be 2**n times'
        self.down_times = 1
        self.N = N
        self.down_embed = DownEmbedBlock(in_channels=in_dim, out_channels=d_model, down_times=self.down_times,
                                         kernel_size=3, stride=2, padding=1, dropout=dropout)
        self.up_embed = UpEmbedBlock(in_channels=in_dim, out_channels=d_model, down_times=self.down_times,
                                     kernel_size=3, stride=2, padding=1, dropout=dropout)
        
        self.pos_encoder = Conv3dPosEmbedding(dim=d_model, dropout=dropout, emb_kernel=3)
        # pos_encoder = Conv3dPosEmbedding(dim=d_model, dropout=dropout, emb_kernel=3)
        # self.pos_encoders = clones(pos_encoder, N)
        attn_layer = SelfAttentionLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout)
        self.layers = clones(attn_layer, N)

        # self.pos_encodes = clones(pos_encode, N)

    def forward(self, x: torch.Tensor):
        '''
        x: Input x size
            [nbatch, channel, height, width, depth]
        Mask:
            [nbatch, 1, height, width, depth]
        '''
        # embedding the orginal size
        x = self.down_embed(x)

        # reshape the position to depth, height, width
        nbatch, _, height, width, depth = x.shape
        x = x.permute((0, 1, 4, 2, 3))
        x = x.flatten(start_dim=2).transpose_(1, 2)

        for i in range(self.N):
            x = self.layers[i](x)
            
            if i == 0:
                x = x.transpose_(1, 2).reshape(nbatch, -1, depth, height, width)
                x = self.pos_encoder(x)
                x = x.flatten(start_dim=2).transpose_(1, 2)
            '''
            x = x.transpose_(1, 2).reshape(nbatch, -1, depth, height, width)
            x = self.pos_encoders[i](x)
            x = x.flatten(start_dim=2).transpose_(1, 2)
            '''
        x = x.transpose_(1, 2).reshape(nbatch, -1, depth, height, width)
        x = x.permute((0, 1, 3, 4, 2))

        # un-embedding the orginal size
        x = self.up_embed(x)
        return x


class UpBlock(nn.Module):
    '''
    Upsampling block for UNet
    Args:
        in_channels: input channel of 3D convolution block
        out_channels: output channel of 3D convolution block
        up_ratio: up sampleing ratio
        kernel_size: the kernel size for convolution
        stride: stride for convolution
        paddding: the padding for convolution
        dropout: dropout ratio
        is_res: whether apply res block or not
        activation: apply relu or leakyrelu activation function
        dropout
    '''
    def __init__(self, in_channels:int, out_channels:int, up_ratio:int = 2, kernel_size:int = 3, stride:Union[int, Tuple[int,...]] = 2,
                 padding: int = 0, dropout:float=None, is_res:bool = True, 
                 activation='leakyrelu'):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding='same')
        # self.norm1 = nn.InstanceNorm3d(num_features=in_channels)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels)

        self.conv2 = nn.Conv3d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding='same')
        # self.norm2 = nn.InstanceNorm3d(num_features=2*out_channels)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels)
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, skip):
        """
        x is the output of the former block
        skip is the output of the same level block during encoding
        """
        x = self.conv1(x)
        # x_up = self.conv1(self.norm1(x_up))
        x = self.norm1(x)
        x = self.activation(x)

        # cat in the channel wise
        # x_cat = self.norm2(torch.cat((x_up, skip), dim=1))
        # x_conv = self.activation(self.conv2(x_cat))
        x = self.conv2(torch.cat((x, skip), dim=1))
        x = self.activation(self.norm2(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):
    '''
    Encoding the input according to the num_layers
    Return the encoded features
    Args:
        num_layers: the channel of each layer
        kernel_size: the size of convolution kernel
        dropout: the dropout ratio
        dim_input: input channel wise
    Output:
        encoded_feature: the final encoded feature
        encoded_feature_list: the encoded features maps,
            in list[num_layers-1](Tensor([batch, channels, heights, widths, depths])) format
    '''
    def __init__(self, num_layers:list, dim_input:int, kernel_size:int=3, dropout:float=None):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.emb_window = 2
        # stride = ((i-1)%2+1)) for z in none-unified sample
        # stride 2
        ''''''
        self.block_list = nn.ModuleList([
            DownBlock(in_channels=num_layers[i-1], out_channels=num_layers[i], 
                      kernel_size=kernel_size, dropout=dropout, 
                      stride=(2, 2, (i-1)%2+1), padding=kernel_size//2) 
                      for i in range(1, len(num_layers))
            ])

        self.input_block = nn.Conv3d(in_channels=dim_input*self.emb_window**2, 
                                     out_channels=self.num_layers[0], 
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding='same')
        self.norm1 = nn.InstanceNorm3d(num_features=self.num_layers[0])
        self.activation1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = windows_embedding(x, kernel_size=self.emb_window)
        x = self.input_block(x)
        x = self.norm1(x)
        x = self.activation1(x)

        encoded_feature_list = []
        for i in range(len(self.num_layers)-1):
            x, inter_block = self.block_list[i](x)
            encoded_feature_list.append(inter_block)

        return x, encoded_feature_list


class Decoder(nn.Module):
    '''
    Decoding the input feature maps according to the num_layers
    Return the segmentation features
    Args:
        num_layers: the channel of each layer, reverse is needed
        kernel_size: the size of convolution kernel
        dim_output: the final output dim
        dropout: the dropout ratio
    Output:
        final_segmentation: final segmentation maps
            in Tensor([batch, dim_output, heights, widths, depths])
    '''
    def __init__(self, num_layers:list, dim_output:int, kernel_size:int=3, dropout:float=None):
        super().__init__()
        self.num_layers = num_layers

        self.block_list = nn.ModuleList([
            UpBlock(in_channels=self.num_layers[-i], out_channels=self.num_layers[-i-1], 
                    kernel_size=kernel_size, 
                    stride=(2, 2, ((len(self.num_layers)-i-1)%2+1)),
                    dropout=dropout) for i in range(1, len(self.num_layers))
        ])
        self.final_block = nn.Conv3d(in_channels=self.num_layers[0], 
                                     out_channels=dim_output, 
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding='same')
    
    def forward(self, x, encoded_list):
        for i in range(len(self.num_layers)-1):
            x = self.block_list[i](x, encoded_list[-i-1])
        final_segmentation = self.final_block(x)
        final_segmentation = F.softmax(final_segmentation, dim=1)
        return final_segmentation


class ConnectBridge(nn.Module):
    '''
    Here is used to define the connection of final encoded feature maps
    return should be the same size with input
    Args:
        d_model: the dimension for the model, use the bottle layer
        nhead: head for multihead attention
        N: attention repeated times 
    Output:
        return the output connection feature maps in the bottle layer
    '''
    def __init__(self, d_model:int, nhead:int, dropout:float=0.2, N:int = 8):
        super(ConnectBridge, self).__init__()
        self.transformer = PosAttention3DBlock(in_dim=d_model,
                                               d_model=d_model,
                                               nhead=nhead,
                                               dropout=dropout,
                                               N=N)
    
    def forward(self, x):
        """
        Input tensor size [n_batch, channels, height, width, depth]
        """
        return self.transformer(x)


class ROIBridge(nn.Module):
    def __init__(self, in_dim:int, d_model:int, nhead:int, 
                 dropout:float=0.2, N:int = 8,
                 roi_size:int =32, mask_threshold:float=0.5):
        '''
        Here is used to define the connection of skip encoded feature maps
        The transformer block will be only applied on the selected roi region
        return should be the same size with input
        Args:
            d_model: the dimension for the model, use the layer length
            nhead: head for multihead attention
            N: attention repeated times
            roi_size: the size for roi region (x, y plane)
            z_roi_size: the size for roi region in depth level
        Output:
            return the output connection feature maps in the bottle layer
        '''
        super().__init__()
        self.transformer = EmbedAttention3DBlock(in_dim=in_dim,
                                                 d_model=d_model,
                                                 nhead=nhead,
                                                 dropout=dropout,
                                                 N=N)

        self.roi_size = roi_size
        self.h_roi_size = roi_size
        self.w_roi_size = int(roi_size * 0.6)
        # self.w_roi_size = int(roi_size * 1)
        self.eval_roi_size = int(1.2*roi_size)
        self.eval_h_roi_size = self.eval_roi_size
        self.eval_w_roi_size = int(self.eval_h_roi_size*0.6)
        # self.eval_w_roi_size = int(self.eval_h_roi_size*1)
        
        self.min_value = 0.5
        '''
        self.roi_size = 2*roi_size
        self.z_roi_size = 2*z_roi_size
        self.eval_roi_size = int(1.2*roi_size)
        self.eval_z_roi_size = int(1.2*z_roi_size)
        '''
        self.min_h_roi = self.eval_roi_size // 2
        self.min_w_roi = self.eval_w_roi_size // 2
        self.mask_threshold = mask_threshold
    
    def forward(self, x, mask):
        """
        Input tensor size 
            x: [n_batch, channels, heights, widths, depths]
            mask: [n_batch, 1, heights, widths, depths]
            solid: [n_batch, 1]
        """
        # x = x * (mask_d + x_attn) * 0.5
        # x = x * mask
        # !Compare with pure spatial attention here
        '''
        x_stack = torch.cat([x, mask], dim=1)
        x_attn = torch.sigmoid(self.spatial_attention(x_stack))
        x = x * x_attn
        
        if not self.training:
            x_attn = torch.sigmoid(self.spatial_attention(x))
            return x, x_attn
        '''
        # binary_mask = mask_d > (self.mask_threshold/5)
        # binary_mask = mask >= (self.mask_threshold)
        with torch.no_grad():
            boundary_box = self.get_mask_boundary2(mask >= (self.mask_threshold))
        # boundary_box = self.get_mask_boundary2(mask_d)

        # clip the origin feature map according to the mask
        temp_roi = self.roi_alignment2(x=x, boundary_box=boundary_box)
        
        # encoded_feature = self.transformer(x=temp_roi, mask=temp_binary_mask)
        encoded_feature = self.transformer(x=temp_roi)
        # transfer back to [n_batch, channels, roi_size, roi_size, roi_size]

        # map the clipped region back to the raw image size
        temp_x = self.post_processing2(x=x, roi=encoded_feature, boundary_box=boundary_box)
        # test using res-structure to help with training
        # x = x + mask
        # x_stack = torch.cat([x, mask], dim=1)
        x = temp_x
        return x

    def get_mask_boundary(self, mask):
        """
        To get the spatial mask of the output
        Input args:
            x size:
                [n_batch, 1, heights, widths, depths]
        Output x size: note here for boundary box should be 6
            [n_batch, boundary_box]
        """
        # size of mask_to box output
        #   [N, boxes_dim (6)]
        n_batch = mask.size(0)
        device = mask.device
        mask_d = mask.detach()
        height, width, depth = mask_d.size(-3), mask_d.size(-2), mask_d.size(-1)
        # transform the format for roi alignment
        #   [N, 6], each index is one box for image
        boundary_box = torch.zeros((n_batch, 6), device=device, dtype=torch.float)
        for index in range(n_batch):
            x, y, z = torch.where(mask_d[index, 0] != 0)
            x, y, z = x.to(dtype=torch.float), y.to(dtype=torch.float), z.to(dtype=torch.float)
            if x.numel() >= 4:
                boundary_box[index, 0] = torch.min(x)
                boundary_box[index, 1] = torch.min(y)
                boundary_box[index, 2] = torch.min(z)
                boundary_box[index, 3] = torch.max(x)
                boundary_box[index, 4] = torch.max(y)
                boundary_box[index, 5] = torch.max(z)

                heigh_center = torch.mean(x)
                height_size = (boundary_box[index, 3] - boundary_box[index, 0])
                width_center = torch.mean(y)
                width_size = (boundary_box[index, 4] - boundary_box[index, 1])
                depth_center = torch.mean(z)
                depth_size = (boundary_box[index, 5] - boundary_box[index, 2])

                if height_size<self.roi_size:
                    boundary_box[index, 0] = torch.maximum(heigh_center-self.min_roi/2, 
                                                           torch.tensor(0, device=device))
                    boundary_box[index, 3] = torch.minimum(heigh_center+self.min_roi/2,
                                                           torch.tensor(height, device=device))

                if width_size<self.roi_size:
                    boundary_box[index, 1] = torch.maximum(width_center-self.min_roi/2, 
                                                           torch.tensor(0, device=device))
                    boundary_box[index, 4] = torch.minimum(width_center+self.min_roi/2,
                                                           torch.tensor(width, device=device))
                
                if depth_size<self.z_roi_size:
                    boundary_box[index, 2] = torch.maximum(depth_center-self.min_z_roi/2, 
                                                           torch.tensor(0, device=device))
                    boundary_box[index, 5] = torch.minimum(depth_center+self.min_z_roi/2,
                                                           torch.tensor(depth, device=device))

            else:
                boundary_box[index, 0] = (height - self.min_roi) / 2
                boundary_box[index, 1] = (width - self.min_roi) / 2
                boundary_box[index, 2] = (depth - self.min_z_roi) / 2
                boundary_box[index, 3] = (height + self.min_roi) / 2
                boundary_box[index, 4] = (width + self.min_roi) / 2
                boundary_box[index, 5] = (depth + self.min_z_roi) / 2

        return boundary_box
    
    def get_mask_boundary2(self, mask):
        """
        To get the spatial mask of the output
        Input args:
            x size:
                [n_batch, 1, heights, widths, depths]
        Output x size: note here for boundary box should be 6
            [n_batch, boundary_box]
        """
        # size of mask_to box output
        #   [N, boxes_dim (6)]
        n_batch = mask.size(0)
        device = mask.device
        height, width, depth = mask.size(-3), mask.size(-2), mask.size(-1)
        # transform the format for roi alignment
        #   [N, 6+1], each index is one box for image, 6 in total,
        #   last indicate this is solid volume or not
        mask_frame_x = torch.sum(mask, dim=(3, 4))
        mask_frame_y = torch.sum(mask, dim=(2, 4))

        boundary_box = torch.zeros((n_batch, 6), device=device, dtype=torch.float)
        for index in range(n_batch):
            boundary_box[index, 0], boundary_box[index, 3], heigh_center = get_min_max_indice(mask_frame_x[index].squeeze_())
            boundary_box[index, 1], boundary_box[index, 4], width_center = get_min_max_indice(mask_frame_y[index].squeeze_())
            boundary_box[index, 2], boundary_box[index, 5] = 0, depth-1

            height_size = (boundary_box[index, 3] - boundary_box[index, 0])
            width_size = (boundary_box[index, 4] - boundary_box[index, 1])

            if height_size<self.min_h_roi:
                boundary_box[index, 0] = torch.maximum(heigh_center-self.min_h_roi/2, 
                                                        torch.tensor(0, device=device))
                boundary_box[index, 3] = torch.minimum(heigh_center+self.min_h_roi/2,
                                                        torch.tensor(height, device=device))

            if height_size>(height -self.min_h_roi):
                boundary_box[index, 0] = torch.maximum(heigh_center-(height-self.min_h_roi)/2, 
                                                        torch.tensor(0, device=device))
                boundary_box[index, 3] = torch.minimum(heigh_center+(height-self.min_h_roi)/2,
                                                        torch.tensor(height, device=device))

            if width_size<self.min_w_roi:
                boundary_box[index, 1] = torch.maximum(width_center-self.min_w_roi/2, 
                                                        torch.tensor(0, device=device))
                boundary_box[index, 4] = torch.minimum(width_center+self.min_w_roi/2,
                                                        torch.tensor(width, device=device))
            if width_size>(width-self.min_w_roi):
                boundary_box[index, 1] = torch.maximum(width_center-(width-self.min_w_roi)/2, 
                                                        torch.tensor(0, device=device))
                boundary_box[index, 4] = torch.minimum(width_center+(width-self.min_w_roi)/2,
                                                        torch.tensor(width, device=device))

        return boundary_box


    def get_mask_boundary3(self, mask):
        """
        To get the spatial mask of the output
        Input args:
            x size:
                [n_batch, 1, heights, widths, depths]
        Output x size: note here for boundary box should be 6
            [n_batch, boundary_box]
        """
        # size of mask_to box output
        #   [N, boxes_dim (6)]
        n_batch = mask.size(0)
        device = mask.device
        height, width, depth = mask.size(-3), mask.size(-2), mask.size(-1)
        # transform the format for roi alignment
        #   [N, 6+1], each index is one box for image, 6 in total,
        #   last indicate this is solid volume or not
        mask_frame_x = torch.sum(mask, dim=(3, 4))
        mask_frame_y = torch.sum(mask, dim=(2, 4))

        boundary_box = torch.zeros((n_batch, 6), device=device, dtype=torch.float)
        for index in range(n_batch):
            boundary_box[index, 0], boundary_box[index, 3], heigh_center = get_min_max_indice(mask_frame_x[index].squeeze_())
            boundary_box[index, 1], boundary_box[index, 4], width_center = get_min_max_indice(mask_frame_y[index].squeeze_())
            boundary_box[index, 2], boundary_box[index, 5] = 0, depth-1

            height_size = (boundary_box[index, 3] - boundary_box[index, 0])
            width_size = (boundary_box[index, 4] - boundary_box[index, 1])

            if height_size<self.h_roi_size:
                boundary_box[index, 0] = torch.maximum(heigh_center-self.h_roi_size/2, 
                                                        torch.tensor(0, device=device))
                boundary_box[index, 3] = torch.minimum(heigh_center+self.h_roi_size/2,
                                                        torch.tensor(height, device=device))

            if height_size>(height -self.min_h_roi):
                boundary_box[index, 0] = torch.maximum(heigh_center-(height-self.min_h_roi)/2, 
                                                        torch.tensor(0, device=device))
                boundary_box[index, 3] = torch.minimum(heigh_center+(height-self.min_h_roi)/2,
                                                        torch.tensor(height, device=device))

            if width_size<self.w_roi_size:
                boundary_box[index, 1] = torch.maximum(width_center-self.w_roi_size/2, 
                                                        torch.tensor(0, device=device))
                boundary_box[index, 4] = torch.minimum(width_center+self.w_roi_size/2,
                                                        torch.tensor(width, device=device))
            if width_size>(width-self.min_w_roi):
                boundary_box[index, 1] = torch.maximum(width_center-(width-self.min_w_roi)/2, 
                                                        torch.tensor(0, device=device))
                boundary_box[index, 4] = torch.minimum(width_center+(width-self.min_w_roi)/2,
                                                        torch.tensor(width, device=device))

        return boundary_box

    def roi_alignment(self, x, boundary_box):
        """
        To align the feature map according to the boxes
        Input args:
            x size:
                [n_batch, channel, height, width, depth]
            boundary_box:
                the boundary_box, [n_batch, 6]
            roi_size:
                the aligned roi size
            z_roi_size:
                the aligned roi size on depth
        Output x size:
            [n_batch, channel, roi_size, roi_size, z_roi_size, z_roi_size]
        """
        device = x.device
        n_batch, _, h, w, d = x.shape
        # change to real distance
        h, w, d = h-1, w-1, d-1
        '''
        print('before roi_alignment')
        print(torch.any(torch.isnan(x)))
        '''
        x0_int, y0_int, z0_int = 0, 0, 0
        x1_int, y1_int, z1_int = self.roi_size, self.roi_size, self.z_roi_size
        x0, y0, z0, x1, y1, z1 = torch.split(boundary_box, 1, dim=1)
        img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32)
        img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32)
        img_z = torch.arange(z0_int, z1_int, device=device, dtype=torch.float32)
        
        img_x = 2 * img_x/x1_int * (x1-x0)/h + 2 * x0/h - 1
        img_y = 2 * img_y/y1_int * (y1-y0)/w + 2 * y0/w - 1
        img_z = 2 * img_z/z1_int * (z1-z0)/d + 2 * z0/d - 1
        # img_x, img_y have shapes (n_batch, roi_size), (n_batch, roi_size), (n_batch, z_roi_size)

        gx = img_x[:, :, None, None].expand(n_batch, img_x.size(1), img_y.size(1), img_z.size(1))
        gy = img_y[:, None, :, None].expand(n_batch, img_x.size(1), img_y.size(1), img_z.size(1))
        gz = img_z[:, None, None, :].expand(n_batch, img_x.size(1), img_y.size(1), img_z.size(1))
        grid = torch.stack([gz, gy, gx], dim=-1)
        '''
        print('grid check')
        print(torch.any(torch.isnan(grid)))
        '''
        roi = F.grid_sample(x, grid.to(x.dtype), align_corners=True)
        '''
        print('after roi_alignment')
        print(torch.any(torch.isnan(roi)))
        
        if torch.any(torch.isnan(roi)):
            print(torch.any(torch.isnan(x)))
            print(torch.any(torch.isnan(roi)))
            print(roi)
        '''
        return roi

    def roi_alignment2(self, x, boundary_box):
        """
        To align the feature map according to the boxes
            Keep the depth direction unchanged
        Input args:
            x size:
                [n_batch, channel, height, width, depth]
            boundary_box:
                the boundary_box, [n_batch, 6]
        Output x size:
            [n_batch, channel, eval_roi_size, eval_roi_size, depth]
        """
        device = x.device
        n_batch, channel, h, w, d = x.shape
        # change to real distance
        #   h, w = h-1, w-1
        '''
        print('before roi_alignment')
        print(torch.any(torch.isnan(x)))
        '''
        x0, y0,_, x1, y1, _ = torch.split(boundary_box, 1, dim=1)

        img_x = get_transfer_index(x0, x1, h-1, self.h_roi_size, self.eval_h_roi_size, device=device)
        img_y = get_transfer_index(y0, y1, w-1, self.w_roi_size, self.eval_w_roi_size, device=device)

        gx = img_x[:, None, :, None].expand(n_batch, d, img_x.size(1), img_y.size(1))
        gy = img_y[:, None, None, :].expand(n_batch, d, img_x.size(1), img_y.size(1))
        gx = gx.flatten(0, 1)
        gy = gy.flatten(0, 1)
        grid = torch.stack([gy, gx], dim=-1)
        '''
        print('grid check')
        print(torch.any(torch.isnan(grid)))
        
        if torch.any(torch.isnan(grid)):
            print('x range')
            print(x0, x1)
            print('y range')
            print(y0, y1)
            print('grid nan lol')
            print(grid.shape)
            print(torch.max(grid))
            print('grid number')
            print(img_x)
            print(img_y)
        '''
        # interpolate in x, y direction
        x = x.permute(0, 4, 1, 2, 3).flatten(0, 1)
        # out roi size: [n_batch*depth, channel, eval_roi_size, eval_roi_size]
        roi = F.grid_sample(x, grid.to(x.dtype), align_corners=True)
        
        roi = roi.reshape(n_batch, d, channel, self.eval_h_roi_size, self.eval_w_roi_size)
        # [n_batch, channel, eval_roi_size, eval_roi_size, depth]
        roi = roi.permute(0, 2, 3, 4, 1)
        return roi

    def post_processing(self, x, roi, boundary_box):
        '''
        To restore the transformed region back to the raw image
        Input args:
            x: the original x
                [n_batch, channel, height, width, depth]
            roi: the selected roi region after transform
                [n_batch, channel, roi_size, roi_size, z_roi_size]
            mask: the mask for the roi region
                [n_batch, 1, height, width. depth]
            boundary_box: the boundary box
                [n_batch, boxes]
        Return args:
            x: the same size with input x
                [n_batch, channel, height, width, depth]
        '''
        device = x.device
        n_batch, _, x1_int, y1_int, z1_int = x.shape
        x0_int, y0_int, z0_int = 0, 0, 0
        x0, y0, z0, x1, y1, z1 = torch.split(boundary_box, 1, dim=1)
        img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32)
        img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32)
        img_z = torch.arange(z0_int, z1_int, device=device, dtype=torch.float32)
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1
        img_z = (img_z - z0) / (z1 - z0) * 2 - 1
        # img_x, img_y have shapes (n_batch, height), (n_batch, width)
        
        gx = img_x[:, :, None, None].expand(n_batch, img_x.size(1), img_y.size(1), img_z.size(1))
        gy = img_y[:, None, :, None].expand(n_batch, img_x.size(1), img_y.size(1), img_z.size(1))
        gz = img_z[:, None, None, :].expand(n_batch, img_x.size(1), img_y.size(1), img_z.size(1))
        grid = torch.stack([gz, gy, gx], dim=-1)

        img_masks = F.grid_sample(roi, grid.to(roi.dtype), align_corners=True)
        # ! Try without res structure here
        x = img_masks
        # x = img_masks + x
        return x

    def post_processing2(self, x, roi, boundary_box):
        '''
        To restore the transformed region back to the raw image
        Input args:
            x: the original x
                [n_batch, channel, height, width, depth]
            roi: the selected roi region after transform
                [n_batch, channel, eval_roi_size, eval_roi_size, depth]
            mask: the mask for the roi region
                [n_batch, 1, height, width. depth]
            boundary_box: the boundary box
                [n_batch, boxes]
        Return args:
            x: the same size with input x
                [n_batch, channel, height, width, depth]
        '''
        device = x.device
        n_batch, channel, h, w, d = x.shape
        # change to real distance
        # h, w = h-1, w-1
        x0, y0, _, x1, y1, _ = torch.split(boundary_box, 1, dim=1)
        img_x = get_transfer_back_index(x0, x1, h-1, self.h_roi_size, self.eval_h_roi_size, device)
        img_y = get_transfer_back_index(y0, y1, w-1, self.w_roi_size, self.eval_w_roi_size, device)

        gx = img_x[:, None, :, None].expand(n_batch, d, img_x.size(1), img_y.size(1))
        gy = img_y[:, None, None, :].expand(n_batch, d, img_x.size(1), img_y.size(1))
        gx = gx.flatten(0, 1)
        gy = gy.flatten(0, 1)
        grid = torch.stack([gy, gx], dim=-1)

        # reshape roi in depth direction
        roi = roi.permute(0, 4, 1, 2, 3).flatten(0, 1)
        x = F.grid_sample(roi, grid.to(roi.dtype), align_corners=True)
        # ! Try without res structure here
        x = x.reshape(n_batch, d, channel, h, w)
        # [n_batch, channel, eval_roi_size, eval_roi_size, depth]
        x = x.permute(0, 2, 3, 4, 1)
        return x

    def eval_forward(self, x, mask):
        '''
        For evaluation procedure, the attention is calculate on pixel level
        rather than using resizing
        Input args:
            x size:
                [n_batch, channel, height, width, depth]
        Output args:
            out size:
                [n_batch, channel, height, width, depth]
        '''
        # size of mask_to box output
        #   [N, boxes_dim (6)]
        n_batch, channel, _, _, _ = x.shape

        binary_mask = mask > self.mask_threshold
        boundary_box = self.get_mask_boundary(binary_mask)
        # clip the origin feature map according to the mask
        temp_roi = self.roi_alignment(x=x, boundary_box=boundary_box,
                                      roi_size=self.eval_roi_size,
                                      z_roi_size=self.eval_z_roi_size)
        temp_mask = self.roi_alignment(x=mask, boundary_box=boundary_box,
                                       roi_size=self.eval_roi_size,
                                       z_roi_size=self.eval_z_roi_size)
        temp_binary_mask = temp_mask > self.mask_threshold
        # out_roi = torch.zeros_like(temp_roi, device=device)
        # reshape the roi region for transformer block
        temp_roi_flatten = temp_roi.flatten(2).transpose(1, 2)
        temp_mask_flatten = temp_binary_mask.flatten(2).transpose(1, 2)

        encoded_feature = self.transformer(x=temp_roi_flatten,
                                           mask=temp_mask_flatten)
        '''
        encoded_feature = self.transformer(x=temp_roi_flatten)
        '''
        # transfer back to [n_batch, channels, roi_size, roi_size, roi_size]
        out_roi = encoded_feature.transpose(1, 2).reshape((n_batch, -1, self.eval_roi_size, self.eval_roi_size, self.eval_z_roi_size))
        # map the clipped region back to the raw image size

        '''
        for i in range(n_batch):
            single_binary_mask = temp_binary_mask[i].unsqueeze_(0)
            single_image = temp_roi[i].unsqueeze_(0)
            single_roi = torch.masked_select(input=single_image,
                                             mask=single_binary_mask)
            single_roi = single_roi.reshape(1, channel, -1)
            # reshape the roi region for transformer block
            # size: [1, roi_lens, channels]
            single_roi_flatten = single_roi.transpose(1, 2)
            encoded_feature = self.transformer(x=single_roi_flatten)
            # transfer back to [1, channels, roi_lens]
            # map the clipped region back to the raw image size
            temp_out = out_roi[i].masked_scatter(mask=temp_binary_mask,
                                                 source=encoded_feature)
            out_roi[i] = temp_out.squeeze(0)
            # out[i] = x[i] + temp_out.squeeze(0)
        '''
        out = self.post_processing(x=x, roi=out_roi, boundary_box=boundary_box)
        return out


class InitialBridge(nn.Module):
    def __init__(self, d_model):
        '''
        Here is used to define the connection of skip encoded feature maps
        without transformer for initial layer
        Args:
            d_model: the dimension of the model
        Output:
            return the output connection feature maps in the bottle layer
        '''
        super().__init__()

    def forward(self, x, mask):
        """
        Input tensor size 
            x: [n_batch, channels, heights, widths, depths]
            mask: [n_batch, 1, heights, widths, depths]
        """
        # x = x * mask
        return x


class Bridge(nn.Module):
    '''
    Here is used to define the skip connection for each layer
    decode the feature of each block from bottle to up
    Args:
        num_layer: the dimension input for each layer
        d_model: the dimension for the model, use the bottle layer
        nhead: head lens for multihead attention
        N: attention repeated times
        roi_size: the base roi size set for applying transformer
    Output:
        return the encoded feature map for each block
    '''
    def __init__(self, num_layers:list, roi_size:int=16,
                 nhead_lens:int=16, dropout:float=0.2, N:int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.nhead_lens = nhead_lens
        self.dropout = dropout
        self.N = N
        self.roi_size = roi_size
        self.roi_size_list = [self.roi_size*(i+3) for i in range(len(self.num_layers)-1)]
        self.threshold = 0.5

        self.bridge_list = nn.ModuleList([
                            ROIBridge(d_model=self.num_layers[i],
                            nhead=self.num_layers[i]//self.nhead_lens,
                            dropout=self.dropout,
                            N=self.N,
                            roi_size=self.roi_size_list[-i-1],
                            mask_threshold=self.threshold)
                                for i in range(len(num_layers)-1)])
        self.bridge_list.append(ConnectBridge(d_model=self.num_layers[-1],
                            nhead=self.num_layers[-1]//self.nhead_lens,
                            dropout=self.dropout,
                            N=self.N))

        self.mask_conv_list = nn.ModuleList([
                                        nn.Conv3d(in_channels=num_layers[i],
                                                  out_channels=1, kernel_size=1)
                                            for i in range(len(num_layers))])

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', 
                                    align_corners=True)

    def forward(self, x, encoded_list:list):
        '''
        Input x size:
           [n_batch, channels, height, width]
        Input encoded_list:
            note channel, height, width should be different for each layer
            List([n_batch, channels, height, width])
        '''
        trans_list = []
        mask_list = []
        for i in range(len(self.num_layers)):
            if i==0:
                temp_out = self.bridge_list[-i-1](x)
                x = temp_out
                mask = torch.sigmoid(self.mask_conv_list[-i-1](temp_out))
                mask = self.upsample(mask)
                mask_list.append(mask)
            else:
                temp_out, x_attn = self.bridge_list[-i-1](x=encoded_list[-i], mask=mask)
                mask_list[-1] = (x_attn + mask_list[-1]) / 2
                trans_list.append(temp_out)

                if i != (len(self.num_layers) - 1):
                    mask = torch.sigmoid(self.mask_conv_list[-i-1](temp_out))
                    mask = self.upsample(mask)
                    mask_list.append(mask)

        return x, trans_list[::-1], mask_list


class ROIDecoder(nn.Module):
    '''
    Here is used to define the skip connection for each layer
    decode the feature of each block from bottle to up
    Args:
        num_layer: the dimension input for each layer
        dim_output: the output dimension
        kernel_size: the kernel size for decoder
        d_model: the dimension for the model, use the bottle layer
        nhead: head lens for multihead attention
        N: attention repeated times
        roi_size: the base roi size set for applying transformer
    Output:
        return the encoded feature map for each block
    '''
    def __init__(self, num_layers:list, roi_size_list:list, 
                 is_roi_list:list, dim_output:int, kernel_size:int=3,
                 nhead_lens:int=32, dropout:float=0.2, N:int = 8,
                 stride_list=None):
        super().__init__()
        self.num_layers = num_layers
        self.emb_window = 2
        self.nhead_lens = nhead_lens
        self.dropout = dropout
        self.N = N
        self.threshold = 0.5
        self.roi_size_list = roi_size_list
        self.is_roi_list = is_roi_list
        '''
        if stride_list is None:
            self.stride_list = [i%2==0 for i in range(len(num_layers))]
        else:
            self.stride_list = stride_list
        '''
        self.bridge_list = nn.ModuleList([
                            ROIBridge(in_dim=self.num_layers[i],
                                      d_model=min(4*self.num_layers[i], 256),
                                      nhead=min(4*self.num_layers[i], 256)//32,
                                      dropout=self.dropout,
                                      N=self.N,
                                      roi_size=self.roi_size_list[i],
                                      mask_threshold=self.threshold)
                            if self.is_roi_list[i] else InitialBridge(d_model=self.num_layers[i])
                                for i in range(len(num_layers)-1)])
        self.roi_num = len(num_layers) - 1

        self.bridge_list.append(ConnectBridge(d_model=self.num_layers[-1],
                            nhead=self.num_layers[-1]//self.nhead_lens,
                            dropout=self.dropout,
                            N=self.N))
        self.mask_conv_list = nn.ModuleList([
                                        nn.Conv3d(in_channels=num_layers[i],
                                                  out_channels=dim_output,
                                                  kernel_size=kernel_size,
                                                  padding=kernel_size//2)
                                            for i in range(1, len(self.num_layers))])
        
        self.att_conv_list = nn.ModuleList([
                                       SpatialAttention3DBlock(in_channel1=self.num_layers[i-1], 
                                                               in_channel2=self.num_layers[i], 
                                                               inter_channel=self.num_layers[i-1],
                                                               dim_output=1)
                                            for i in range(1, len(self.num_layers))])
        
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', 
                                    align_corners=True)
        # unified upsample for deep z direction (2, 2, 1)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', 
                                    align_corners=True)
        # stride  (i%2+1) for none unified upsample
        self.block_list = nn.ModuleList([
            UpBlock(in_channels=self.num_layers[-i], out_channels=self.num_layers[-i-1], 
                    kernel_size=kernel_size, stride=(2, 2, 2), dropout=dropout)
                        for i in range(1, len(self.num_layers))
        ])

        self.final_block = nn.Conv3d(in_channels=self.num_layers[0], 
                                     out_channels=dim_output*self.emb_window**2, 
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding='same')

    def forward(self, x, encoded_list:list):
        '''
        Input x size:
           [n_batch, channels, height, width, depth]
        Input encoded_list:
            note channel, height, width, depth should be different for each layer
            List([n_batch, channels, height, width, depth])
        '''
        mask_list = []

        for i in range(len(self.num_layers)):
            if i==0:
                x = self.bridge_list[-i-1](x)
            else:
                # transfer mask to foreground mask
                
                if (len(self.num_layers)-i) % 2 == 0:
                    x = self.upsample(x)
                else:
                    x = self.upsample2(x)
                # x = self.upsample(x)
                mask = self.mask_conv_list[-i](x)
                mask = torch.softmax(mask, dim=1)
                mask_list.append(mask)

                attn = self.att_conv_list[-i](x=encoded_list[-i], up=x)
                skip = encoded_list[-i]*attn

                foreground_mask = (1 - mask[:, 0]).unsqueeze_(1)
                temp_out = self.bridge_list[-i-1](x=skip, mask=foreground_mask)

                x = self.block_list[i-1](x, temp_out)

        x = self.final_block(x)
        x = windows_unembedding(x, kernel_size=self.emb_window)
        x = F.softmax(x, dim=1)

        return x, mask_list


class MaskDecoder(nn.Module):
    def __init__(self, num_layers:list, weight_list:Union[None, list]=None,
                 kernel_size:int=3, stride:Union[int, Tuple[int,...]]=2):
        super().__init__()
        self.num_layers = num_layers
        self.n_layers = len(num_layers)
        if weight_list is None:
            weight_list = [1/(self.n_layers-1)] * (self.n_layers-1)

        self.weight_list = weight_list
        self.kernel_size = kernel_size
        self.recons_kernel = 5
        self.stride = stride

        # (2**i, 2**i, 2**(i//2))
        self.up_decoder = nn.ModuleList([nn.Upsample(scale_factor=(2**(i+1), 2**(i+1), 2**(i//2)),
                                                     mode='trilinear',
                                                     align_corners=True)
                                            for i in range(self.n_layers-1)])

    def forward(self, mask_list:list):
        assert len(mask_list) == (self.n_layers-1), 'the layer should be same'

        out_list = [self.up_decoder[-i-1](mask) for i, mask in enumerate(mask_list)]

        return out_list
