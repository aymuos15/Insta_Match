import torch 
import torch.nn as nn

import cupy as cp
from cucim.skimage import measure as cucim_measure

from metrics.legacy_dice import legacy_dice as dice

"""
Lesion-wise Dice loss
"""

def get_connected_components(img, connectivity=None):
    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)
    return labeled_img_torch, num_features

class LesionWiseDiceLoss(nn.Module):
    def forward(self, pred: torch.Tensor, label: torch.Tensor):
        # Remove gradients from pred and label
        pred = pred.detach()
        label = label.detach()

        pred_label_cc, num_pred_lesions = get_connected_components(pred)
        gt_label_cc, num_gt_lesions = get_connected_components(label)

        lesion_dice_scores = 0
        tp = torch.tensor([], device=pred_label_cc.device)

        for gtcomp in range(1, num_gt_lesions + 1):
            gt_tmp = (gt_label_cc == gtcomp)
            intersecting_cc = torch.unique(pred_label_cc[gt_tmp])
            intersecting_cc = intersecting_cc[intersecting_cc != 0]

            if len(intersecting_cc) > 0:
                pred_tmp = torch.zeros_like(pred_label_cc, dtype=torch.float32)
                pred_tmp = torch.where(torch.isin(pred_label_cc, intersecting_cc), torch.tensor(1., device=pred_label_cc.device), pred_tmp)
                dice_score = dice(pred_tmp, gt_tmp)
                lesion_dice_scores += dice_score
                tp = torch.cat([tp, intersecting_cc])
        
        mask = (pred_label_cc != 0) & (~torch.isin(pred_label_cc, tp))
        fp = torch.unique(pred_label_cc[mask], sorted=True)
        fp = fp[fp != 0]

        # Calculate and return the loss
        return 1 - lesion_dice_scores / (num_gt_lesions + len(fp))