import numpy as np
import cc3d
from scipy.ndimage import distance_transform_edt
from surface_metrics.legacy_dist import compute_surface_distances
from surface_metrics.legacy_dist import compute_robust_hausdorff

def RegionDist(pred, gt):
    # Step 1: Create the overlay
    overlay = pred + gt
    overlay[overlay > 0] = 1

    # Step 2: Identify blob centers and label each region
    labeled_array = cc3d.connected_components(overlay)
    num_connected_components = np.max(labeled_array)

    # Step 3: Compute the Euclidean distance transform for each region (vectorized)
    # Create a 4D array where each slice along the 4th dimension is a binary mask for each label
    masks = (labeled_array[..., np.newaxis] == np.arange(1, num_connected_components + 1))
    
    # Compute distances for all masks at once
    distances = distance_transform_edt(~masks)
    
    # Create the region map by finding the index of the minimum distance
    region_map = np.argmin(distances, axis=-1) + 1

    # Step 4: Calculate distance metrics within each region
    dist_scores = []

    for region_label in range(1, num_connected_components + 1):
        region_mask = (region_map == region_label)
        
        # Apply the region mask to both pred and gt
        pred_region = np.zeros_like(pred)
        gt_region = np.zeros_like(gt)
        
        pred_region[region_mask] = pred[region_mask]
        gt_region[region_mask] = gt[region_mask]

        # Compute surface distances for the region
        lesion_dist = compute_surface_distances(pred_region.astype(bool), 
                                              gt_region.astype(bool), 
                                              spacing_mm=(1, 1, 1))
        surface_score = compute_robust_hausdorff(lesion_dist, 95)
        dist_scores.append(surface_score)

    # Calculate the overall distance score as the average of regional distances
    overall_dist = np.mean(dist_scores)

    return overall_dist