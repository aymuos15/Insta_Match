import torch
import torch.nn as nn

import cupy as cp
from cucim.skimage import measure as cucim_measure

class ClusterDiceLoss(nn.Module):
    def __init__(self):
        super(ClusterDiceLoss, self).__init__()
    
    def dice_score(self, pred, gt):
        """Calculate Dice score for a single cluster"""
        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt)
        
        if union == 0:
            return 1.0
        
        return 2 * intersection / union

    def get_connected_components(self, img, connectivity=None):
        img_cupy = cp.asarray(img)
        labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
        return labeled_img, num_features
    
    def forward(self, pred, target):
        # Step 1: Create the overlay
        overlay = pred + target
        overlay = (overlay > 0).float()
        
        # Step 2: Cluster the overlay
        labeled_array, num_features = self.get_connected_components(overlay)
        
        # Convert labeled_array from CuPy to PyTorch tensor
        labeled_array = torch.from_numpy(cp.asnumpy(labeled_array)).to(pred.device)
        
        # Step 3: Calculate Dice scores for each cluster
        dice_scores = []
        
        for cluster in range(1, num_features + 1):
            cluster_mask = (labeled_array == cluster).float()
            
            pred_cluster = torch.logical_and(pred, cluster_mask).float()
            gt_cluster = torch.logical_and(target, cluster_mask).float()
            
            dice_score = self.dice_score(pred_cluster, gt_cluster)
            dice_scores.append(dice_score)
        
        if not dice_scores:  # Handle case when there are no clusters
            return torch.tensor(1.0, device=pred.device)
        
        dice_scores = torch.stack(dice_scores)
        
        # Return loss (1 - mean Dice score to convert from similarity to loss)
        return 1.0 - torch.mean(dice_scores)