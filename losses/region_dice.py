import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
import cupy as cp
from cucim.skimage import measure as cucim_measure

class RegionDiceLoss(nn.Module):
    def __init__(self):
        super(RegionDiceLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_connected_components(self, img):
        img_cupy = cp.asarray(img)
        labeled_img, num_features = cucim_measure.label(img_cupy, return_num=True)
        labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)
        return labeled_img_torch, num_features
    
    def legacy_dice(self, pred, gt):
        intersection = torch.sum(pred * gt)
        dice = (2. * intersection) / (torch.sum(pred) + torch.sum(gt) + 1e-8)
        return dice
    
    def get_regions(self, pred, gt):
        # Step 1: Connected Components for 3D volume
        labeled_array, num_features = self.get_connected_components(gt.cpu())    
        
        # Step 2: Compute distance transform for each 3D region
        distance_map = torch.zeros_like(gt, dtype=torch.float32)
        region_map = torch.zeros_like(gt, dtype=torch.long)
        
        for region_label in range(1, num_features + 1):
            region_mask = (labeled_array == region_label)
            
            # Convert to numpy for distance transform
            region_mask_np = region_mask.cpu().numpy()
            distance = torch.from_numpy(
                distance_transform_edt(~region_mask_np)
            ).to(self.device)
            
            if region_label == 1 or distance_map.max() == 0:
                distance_map = distance
                region_map = region_label * torch.ones_like(gt, dtype=torch.long)
            else:
                update_mask = distance < distance_map
                distance_map[update_mask] = distance[update_mask]
                region_map[update_mask] = region_label
        
        return region_map, num_features
    
    def forward(self, pred, target):

        """
        Args:
            pred (torch.Tensor): Predicted segmentation mask (B, C, D, H, W)
            target (torch.Tensor): Ground truth segmentation mask (B, C, D, H, W)
        Returns:
            torch.Tensor: Region-based Dice loss for 3D volumes
        """

        # Ensure inputs are proper probabilities
        pred = torch.sigmoid(pred) if pred.dim() == 5 else pred
        
        batch_size = pred.size(0)
        losses = []
        
        for b in range(batch_size):
            pred_volume = pred[b].squeeze()  # (D, H, W)
            target_volume = target[b].squeeze()  # (D, H, W)
            
            region_map, num_features = self.get_regions(pred_volume, target_volume)
            
            if num_features == 0:
                # Handle cases with no regions
                losses.append(torch.tensor(1.0, device=self.device))
                continue
                
            region_dice_scores = []
            for region_label in range(1, num_features + 1):
                region_mask = (region_map == region_label)
                pred_region = pred_volume[region_mask]
                target_region = target_volume[region_mask]
                
                dice_score = self.legacy_dice(pred_region, target_region)
                region_dice_scores.append(dice_score)
            
            # Calculate mean Dice score for this volume
            mean_dice = torch.mean(torch.stack(region_dice_scores))
            losses.append(1 - mean_dice)  # Convert to loss
        
        # Return mean loss across batch
        return torch.mean(torch.stack(losses))