import torch 

from overlap_metrics.legacy_dice import legacy_dice as dice

from metrics_utils.gpu_connected_components import get_connected_components

def lesion_wise_dice(pred, gt):

    '''
    This is an optimised version, on torch of Lesion-wise Dice.
    Original Code: https://github.com/rachitsaluja/BraTS-2023-Metrics
    '''

    """
    prediction: torch.Tensor of shape (H, W, D)
    ground truth: torch.Tensor of shape (H, W, D)
    """

    pred_label_cc, num_pred_lesions = get_connected_components(pred)
    gt_label_cc, num_gt_lesions = get_connected_components(gt)

    num_gt_lesions = torch.unique(gt_label_cc[gt_label_cc != 0]).size(0)

    lesion_dice_scores = 0
    tp = torch.tensor([]).cuda()

    for gtcomp in range(1, num_gt_lesions + 1):
        gt_tmp = (gt_label_cc == gtcomp)
        intersecting_cc = torch.unique(pred_label_cc[gt_tmp])
        intersecting_cc = intersecting_cc[intersecting_cc != 0]

        if len(intersecting_cc) > 0:
            pred_tmp = torch.zeros_like(pred_label_cc, dtype=torch.bool)
            pred_tmp[torch.isin(pred_label_cc, intersecting_cc)] = True
            dice_score = dice(pred_tmp, gt_tmp)
            lesion_dice_scores += dice_score
            tp = torch.cat([tp, intersecting_cc])

    mask = (pred_label_cc != 0) & (~torch.isin(pred_label_cc, tp)).cuda()
    fp = torch.unique(pred_label_cc[mask], sorted=True).cuda()
    fp = fp[fp != 0]

    return lesion_dice_scores / (num_gt_lesions + len(fp))