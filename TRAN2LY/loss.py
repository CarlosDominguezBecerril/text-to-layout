import torch
import math

from torch.nn import functional as F

import numpy as np

def bbox_loss(preds, target, mask=None, reduction='mean'):

    """
    Function to calculate the loss of the bounding boxes
    """
    def calculate_wh_loss(preds, target):
        return F.mse_loss(preds[:, 2:], target[:, 2:], reduction="none").sum(1)

    def calculate_xy_loss(preds, target):
        return F.mse_loss(preds[:, :2], target[:, :2], reduction="none").sum(1)

    # Calculate the loss
    wh_loss = calculate_wh_loss(preds, target)
    xy_loss = calculate_xy_loss(preds, target)

    # Apply a mask
    if mask != None:
        wh_loss = wh_loss * mask
        xy_loss = xy_loss * mask

    # Apply the reduction method
    if reduction == 'mean':
        if mask != None:
            wh_loss = torch.sum(wh_loss) / torch.sum(mask)
            xy_loss = torch.sum(xy_loss) / torch.sum(mask)
        else:
            wh_loss = torch.mean(wh_loss)
            xy_loss = torch.mean(xy_loss)

    elif reduction == 'sum':
        wh_loss = torch.sum(wh_loss)
        xy_loss = torch.sum(xy_loss) 
    else:
        raise NotImplementedError

    return wh_loss, xy_loss