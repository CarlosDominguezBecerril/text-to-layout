import torch
import math

from torch.nn import functional as F

import numpy as np

def bbox_loss(preds, target, mask=None, coco_to_img=None, reduction='mean'):

    def width_and_height_difference_loss(preds, target):
        """
        This loss ensures that no matter the position of the prediction and target bounding boxes the height and width maintains.
        """
        # Calculate the height and width of target and prediction tensors
        width_preds, height_preds = preds[:, 2] - preds[:, 0], preds[:, 3] - preds[:, 1]
        width_target, height_target = target[:, 2] - target[:, 0], target[:, 3] - target[:, 1]

        # Calculate the difference in height and width of target and prediction tensors
        # wh_loss = torch.abs(width_preds - width_target) + torch.abs(height_preds - height_target)
        wh_loss = F.mse_loss(torch.transpose(torch.cat((width_preds.unsqueeze(0), height_preds.unsqueeze(0)), dim=0), 0, 1), torch.transpose(torch.cat((width_target.unsqueeze(0), height_target.unsqueeze(0)), dim=0), 0, 1), reduction="none").sum(1)

        return wh_loss

    def center_distance_loss(preds, target):

        y_pred_center, x_pred_center = (preds[:, 3] + preds[:, 1]) / 2, (preds[:, 2] + preds[:, 0]) / 2
        y_target_center, x_target_center = (target[:, 3] + target[:, 1]) / 2, (target[:, 2] + target[:, 0]) / 2

        # center_loss = torch.abs(y_pred_center - y_target_center) + torch.abs(x_pred_center - x_target_center) # Difference in height and width -> evaluar bien con 10 epochs completos
        
        # center_loss = torch.abs(y_pred_center - y_target_center) # Difference in height -> Doesn't work really well.
        # center_loss = torch.sqrt((y_pred_center - y_target_center)**2 + (x_pred_center - x_target_center)**2) # Euclidian distance -> evaluar bien con 10 epochs completos
        # center_loss = torch.abs(preds[:, 1] - target[:, 1]) + torch.abs(preds[:, 0] - target[:, 0]) # top, right corner difference -> evaluar bien con 10 epochs completos
        # center_loss = torch.abs(preds[:, 1] - target[:, 1]) + torch.abs(preds[:, 0] - target[:, 0])

        center_loss = F.mse_loss(torch.transpose(torch.cat((x_pred_center.unsqueeze(0), y_pred_center.unsqueeze(0)), dim=0), 0, 1), torch.transpose(torch.cat((x_target_center.unsqueeze(0), y_target_center.unsqueeze(0)), dim=0), 0, 1), reduction="none").sum(1)

        return center_loss
            
    
    def overlapping_loss(preds, target, coco_to_img, mask):
        if coco_to_img == None:
            return 0

        total_overlap = 0
        for i in range(int(max(coco_to_img).item())+1):
            indexes = np.where(coco_to_img.cpu().numpy()==i)[0]
            for j in range(len(indexes)):
                for k in range(j+1, len(indexes)):
                    if mask[indexes[j]] == 0 or mask[indexes[k]] == 0:
                        continue
                    ov = torch.abs((iou_loss_simple(preds[indexes[j]].unsqueeze(0), preds[indexes[k]].unsqueeze(0)) - iou_loss_simple(target[indexes[j]].unsqueeze(0), target[indexes[k]].unsqueeze(0)))).item() ** 2
                    total_overlap += ov
        
        return total_overlap

    def calculate_wh_loss(preds, target):
        return F.mse_loss(preds[:, 2:], target[:, 2:], reduction="none").sum(1)

    def calculate_xy_loss(preds, target):
        return F.mse_loss(preds[:, :2], target[:, :2], reduction="none").sum(1)

    wh_loss = calculate_wh_loss(preds, target)
    xy_loss = calculate_xy_loss(preds, target)

    # overlapping_loss = overlapping_loss(preds, target, coco_to_img, mask)

    if mask != None:
        wh_loss = wh_loss * mask
        xy_loss = xy_loss * mask

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

def xy_distribution_loss(preds_distribution_xy, preds_class, target_xy, losses):
    total_loss = 0
    classes = preds_class.cpu().numpy()
    different = 0
    for i in np.unique(classes):
        if i <= 3:
            continue
        indexes = np.where(i == classes)[0]
        indexes = torch.Tensor(indexes).long()
        if torch.cuda.is_available():
            indexes = indexes.cuda()
        pred_distribution = torch.index_select(preds_distribution_xy, 0, indexes)
        pred_target = torch.index_select(target_xy, 0, indexes)
        p = losses[i](pred_distribution, pred_target)
        total_loss += p
        different += 1
    if total_loss == 0:
        if torch.cuda.is_available():
            return torch.Tensor(0, requires_grad=True).cuda()
        else:
            return torch.Tensor(0, requires_grad=True)
    else:
        return total_loss.div(different)


def iou_loss_simple(preds, bbox, mask=None, eps=1e-6, reduction='mean'):
    '''
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: loss
    '''
    x1 = torch.max(preds[:, 0], bbox[:, 0])
    y1 = torch.max(preds[:, 1], bbox[:, 1])
    x2 = torch.min(preds[:, 2], bbox[:, 2])
    y2 = torch.min(preds[:, 3], bbox[:, 3])

    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)

    inters = w * h

    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    return (inters / uni).clamp(min=eps)

# https://github.com/miaoshuyu/object-detection-usages

def giou_loss(preds, bbox, mask=None, eps=1e-7, reduction='mean'):
    '''
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)

    # overlap
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps

    # ious
    ious = inters / uni

    ex1 = torch.min(preds[:, 0], bbox[:, 0])
    ey1 = torch.min(preds[:, 1], bbox[:, 1])
    ex2 = torch.max(preds[:, 2], bbox[:, 2])
    ey2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)

    # enclose erea
    enclose = ew * eh + eps

    giou = ious - (enclose - uni) / enclose

    loss = 1 - ious

    if mask != None:
        loss = loss * mask
        
    if reduction == 'mean':
        if mask != None:
            loss = torch.sum(loss) / torch.sum(mask)
        else:
            loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss

def iou_loss(preds, bbox, mask=None, eps=1e-6, reduction='mean'):
    '''
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: loss
    '''
    x1 = torch.max(preds[:, 0], bbox[:, 0])
    y1 = torch.max(preds[:, 1], bbox[:, 1])
    x2 = torch.min(preds[:, 2], bbox[:, 2])
    y2 = torch.min(preds[:, 3], bbox[:, 3])

    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)

    inters = w * h

    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    ious = (inters / uni).clamp(min=eps)
    loss = -ious.log()

    if mask != None:
        loss = loss * mask

    if reduction == 'mean':
        if mask != None:
            loss = torch.sum(loss) / torch.sum(mask)
        else:
            loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss

def ciou_loss(preds, bbox, mask=None, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)

    # inter_diag
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2

    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag

    # calculate v,alpha
    wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
    hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
    wpreds = preds[:, 2] - preds[:, 0] + 1.0
    hpreds = preds[:, 3] - preds[:, 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    ciou_loss = 1 - ciou

    if mask != None:
        loss = ciou_loss * mask

    if reduction == 'mean':
        if mask != None:
            loss = torch.sum(loss) / torch.sum(mask)
        else:
            loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(ciou_loss)
    else:
        raise NotImplementedError
    return loss

def diou_loss(preds, bbox, mask=None, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)

    # inter_diag
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2

    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)

    diou_loss = 1 - diou

    if mask != None:
        loss = diou_loss * mask

    if reduction == 'mean':
        if mask != None:
            loss = torch.sum(loss) / torch.sum(mask)
        else:
            loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(diou_loss)
    else:
        raise NotImplementedError
    return loss