import torch

from metrics_utils.gpu_connected_components import get_connected_components

"""
Cluster-Dice: Kundu et al
"""

"""
prediction: torch.Tensor of shape (H, W, D)
ground truth: torch.Tensor of shape (H, W, D)
"""

def cluster_dice(pred, gt):
    # Step 1: Create the overlay
    overlay = pred + gt
    overlay[overlay > 0] = 1

    # Step 2: Cluster the overlay
    labeled_array, num_features = get_connected_components(overlay)

    # Step 3: Calculate Dice scores for each cluster using vectorized approach
    one_hot = torch.eye(num_features + 1, device=pred.device)[labeled_array]
    pred_expanded = pred.unsqueeze(-1)
    gt_expanded = gt.unsqueeze(-1)

    pred_cluster = torch.logical_and(pred_expanded, one_hot)
    gt_cluster = torch.logical_and(gt_expanded, one_hot)

    intersection = torch.sum(pred_cluster * gt_cluster, dim=(0, 1, 2))
    union = torch.sum(pred_cluster, dim=(0, 1, 2)) + torch.sum(gt_cluster, dim=(0, 1, 2))

    dice_scores = 2 * intersection / (union + 1e-8)
    
    # Exclude the background (index 0) and calculate mean
    return torch.mean(dice_scores[1:])