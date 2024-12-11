import torch
import pandas as pd

from detection_metrics.IoU import iou

from metrics_utils.gpu_connected_components import get_connected_components

def detection_rate(pred, gt, style=None):

    # Initialize lesion counts for the case
    lesion_counts = {'greater_than_10cc': {'gt': 0, 'pred': 0},
                     '5_to_10cc': {'gt': 0, 'pred': 0},
                     'less_than_5cc': {'gt': 0, 'pred': 0}}

    # Define IoU thresholds
    iou_thresholds = [0.00001, 0.1, 0.25, 0.5, 0.75, 1.0]

    labeled_img, num_features_gt = get_connected_components(gt)
    pred_labeled_img, num_features_pred = get_connected_components(pred)    

    # Initialize the master table DataFrame
    master_table = pd.DataFrame(columns=[
        'Threshold', 
        'Lesion_ID', 'Lesion_Size', 'Size_Category', 'IoU', 'Overlap', 
    ])

    # Iterate through ground truth lesions
    for feature in range(1, num_features_gt + 1):
        component_size = (labeled_img == feature).sum().item()

        # Assign size category based on component size
        if component_size > 1000:
            size_category = 'greater_than_10cc'
        elif 25 <= component_size <= 1000:
            size_category = '5_to_10cc'
        else:
            size_category = 'less_than_5cc'

        # Increment ground truth lesion count for this size category
        lesion_counts[size_category]['gt'] += 1

        # Get the ground truth mask for the current GT component
        gt_mask = (labeled_img == feature)

        # Identify the prediction labels that overlap with the GT component
        overlapping_pred_labels = torch.unique(pred_labeled_img[gt_mask > 0])

        # Remove background label (0)
        overlapping_pred_labels = overlapping_pred_labels[overlapping_pred_labels > 0]

        # If a corresponding prediction component is found, extract its mask
        if overlapping_pred_labels is not None:
            pred_mask = torch.isin(pred_labeled_img, overlapping_pred_labels)
        else:
            # If no overlapping prediction is found, set pred_mask to be an empty mask of the same shape
            pred_mask = torch.zeros_like(gt_mask)

        for threshold in iou_thresholds:
            score = iou(gt_mask, pred_mask)
            score = score.cpu().numpy()
            overlap = score > threshold

            # Add row to master table
            master_table = pd.concat(
                [master_table, pd.DataFrame({
                    'Trainer' : ['Trainer'],
                    'Threshold': [threshold],
                    'Lesion_ID': [feature],
                    'Lesion_Size': [component_size],
                    'Size_Category': [size_category],
                    'IoU': [score],
                    'Overlap': [overlap],
                })],
                ignore_index=True
            )

    # Iterate through predicted lesions and update lesion counts for predicted size categories
    for feature in range(1, num_features_pred + 1):
        pred_component_size = (pred_labeled_img == feature).sum().item()

        # Assign size category for prediction
        if pred_component_size > 1000:
            pred_size_category = 'greater_than_10cc'
        elif 25 <= pred_component_size <= 1000:
            pred_size_category = '5_to_10cc'
        else:
            pred_size_category = 'less_than_5cc'

        # Increment predicted lesion count for this size category
        lesion_counts[pred_size_category]['pred'] += 1

    # return master_table
    print(master_table)