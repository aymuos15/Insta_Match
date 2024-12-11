import numpy as np
import cc3d

# from metrics.legacy_dice import dice
from surface_metrics.legacy_dist import compute_surface_distances
from surface_metrics.legacy_dist import compute_robust_hausdorff

#! Need to fix this base. Maybe simply running compute_surface_distances on the clusters may not work.

def ClusterDist(pred, gt):
    # Step 1: Create the overlay
    overlay = pred + gt
    overlay[overlay > 0] = 1

    # Step 2: Cluster the overlay
    labeled_array = cc3d.connected_components(overlay)
    num_features = np.max(np.unique(labeled_array))

    # Step 3: Calculate Dice scores for each cluster
    dist_scores = []

    for cluster in range(1, num_features + 1):        
    
        cluster_mask = labeled_array == cluster
        
        pred_cluster = np.logical_and(pred, cluster_mask)
        gt_cluster = np.logical_and(gt, cluster_mask)
        
        # dice_score = dice(pred_cluster, gt_cluster)
        lesion_dist = (compute_surface_distances(pred_cluster, gt_cluster, (1, 1, 1)))
        surface_score = compute_robust_hausdorff(lesion_dist, 95)
        dist_scores.append(surface_score)
    
    # Calculate and return the mean of Dice scores
    return np.mean(dist_scores)