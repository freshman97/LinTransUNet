from turtle import forward
import torch
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np


def dis_calculation(predict:torch.Tensor, target: torch.Tensor, eps:float=1e-5):
    '''
    Calculate the one dimension distance
        predict: [N, H]
        target: [N, H]
    '''
    n_length = predict.shape[-1]

    dist_pred = torch.cumsum(predict, dim=-1) / (torch.sum(predict, dim=-1, keepdim=True) + eps)
    dist_target = torch.cumsum(target, dim=-1) / (torch.sum(target, dim=-1, keepdim=True) + eps)
    dim_loss = torch.sum(torch.abs(dist_pred-dist_target)) / torch.sqrt(n_length)

    return dim_loss


class LocalizationLoss(nn.Module):
    '''
    The localization loss using one dimension wasserstein distance for image segmentation
    Background is not calculated
    Args:
        smooth: the smooth value
    '''
    def __init__(self, smooth:float=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict:torch.Tensor, target: torch.Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        n_channel = predict.shape[1]
        shp_x = predict.shape
        shp_y = target.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(predict.shape, target.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = target
            else:
                target = target.long()
                y_onehot = torch.zeros(shp_x)
                if predict.device.type == "cuda":
                    y_onehot = y_onehot.cuda(predict.device.index)
                y_onehot.scatter_(1, target, 1)

        for i in range(1, n_channel):
            if i == 1:
                dim_predict = predict[:, i].flatten(2)
                dim_label = y_onehot[:, i].flatten(2)
                dim_predict = torch.sum(dim_predict, dim=-1)
                dim_label = torch.sum(dim_label, dim=-1)
                loss = dis_calculation(predict=dim_predict, target=dim_label)
            else:
                dim_predict = predict[:, i].transpose(1, (i+1))
                dim_label = y_onehot[:, i].transpose(1, (i+1))
                dim_predict = predict[:, i].flatten(2)
                dim_label = y_onehot[:, i].flatten(2)
                dim_predict = torch.sum(dim_predict, dim=-1)
                dim_label = torch.sum(dim_label, dim=-1)
                loss = loss + dis_calculation(predict=dim_predict, target=dim_label)

        return loss
