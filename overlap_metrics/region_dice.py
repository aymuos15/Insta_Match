import torch
from scipy.ndimage import distance_transform_edt

from overlap_metrics.legacy_dice import legacy_dice as dice

from metrics_utils.gpu_connected_components import get_connected_components

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Implementation of CC-Metrics: https://arxiv.org/pdf/2410.18684
'''

"""
connected component of prediction: torch.Tensor of shape (H, W, D)
connected component of ground truth: torch.Tensor of shape (H, W, D)
"""

def get_gt_regions(gt):
    # Step 1: Connected Components (using CPU as cc3d requires numpy)
    labeled_array, num_features = get_connected_components(gt)    
    
    # Step 2: Compute distance transform for each region
    distance_map = torch.zeros_like(gt, dtype=torch.float32)
    region_map = torch.zeros_like(gt, dtype=torch.long)
    
    for region_label in range(1, num_features + 1):
        # Create region mask
        region_mask = (labeled_array == region_label)
        
        # Convert to numpy for distance transform, then back to torch
        # (since PyTorch doesn't have a direct equivalent of distance_transform_edt)
        #! Need to integrate: https://github.com/moyiliyi/GPU-Accelerated-Boundary-Losses-for-Medical-Segmentation
        region_mask_np = region_mask.cpu().numpy()
        distance = torch.from_numpy(
            distance_transform_edt(~region_mask_np)
        ).to(device)
        
        if region_label == 1 or distance_map.max() == 0:
            distance_map = distance
            region_map = region_label * torch.ones_like(gt, dtype=torch.long)
        else:
            update_mask = distance < distance_map
            distance_map[update_mask] = distance[update_mask]
            region_map[update_mask] = region_label
    
    return region_map, num_features

def region_dice(pred, gt):
    region_map, num_features = get_gt_regions(gt)
    
    # Initialize a tensor to store dice scores
    dice_scores = torch.zeros(num_features)
    
    for region_label in range(1, num_features + 1):
        region_mask = (region_map == region_label)
        pred_region = pred[region_mask]
        gt_region = gt[region_mask]
        
        dice_score = dice(pred_region, gt_region)
        # Subtract 1 from region_label since tensor is 0-indexed
        dice_scores[region_label - 1] = dice_score
    
    # Calculate overall Dice score
    overall_dice = torch.mean(dice_scores)
    
    return overall_dice