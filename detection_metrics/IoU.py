import torch

def iou(gt_mask, pred_mask):
    intersection = torch.logical_and(gt_mask, pred_mask)
    union = torch.logical_or(gt_mask, pred_mask)
    iou = torch.sum(intersection) / torch.sum(union) if torch.sum(union) > 0 else 0
    return iou