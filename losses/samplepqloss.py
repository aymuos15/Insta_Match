# import torch
# import torch.nn as nn 
# import torch.autograd as autograd

# #todo: 
# #taken from: https://github.com/aabbas90/COPS/blob/camera_ready/affinityNet/panoptic_affinity/losses.py

# def iou_batch_3d(pred, target, weight, pixel_dims):
#     eps = 1e-1
#     intersection_dense = pred * target
#     intersection = (intersection_dense * weight).sum(pixel_dims)
#     union = ((pred + target - intersection_dense) * weight).sum(pixel_dims)
#     iou = (intersection + eps) / (union + eps)
#     return iou

# class PanopticQualityLoss3D(nn.Module):
#     def __init__(self, thing_ids, eps=1e-1):
#         super().__init__()
#         self.thing_ids = thing_ids
#         self.eps = eps

#     def forward(self, pan_pred_batch, pan_gt_batch, category_indices_batch, weights, foreground_prob, similarity_function):
#         def soft_threshold(input, threshold = 0.5, beta = 4.0):
#             assert(threshold == 0.5) # Only works for 0.5
#             # return 1.0 / (1.0 + torch.pow(input / (1.0 - input), -beta))
#             return torch.pow(input, beta) / (torch.pow(input, beta) + torch.pow(1.0 - input, beta))

#         def sigmoid_threshold(input, threshold = 0.5, scalar = 15.0):
#             return torch.sigmoid(scalar * (input - threshold))

#         ious = similarity_function(pan_pred_batch, pan_gt_batch, weights.unsqueeze(1), [2, 3, 4])
#         true_probability = soft_threshold(ious, threshold=0.5)
#         false_probability = 1.0 - true_probability

#         gt_non_zero_mask = torch.any(torch.any(pan_gt_batch > 0, 2), 2).to(torch.float32)
#         # Populate all categories present in the whole batch to compute class averaged loss.
#         batch_size = ious.shape[0]
#         all_cats = set()
#         for b in range(batch_size):
#             current_cats = category_indices_batch[b].keys()
#             all_cats.update(list(current_cats))
        
#         full_pq = 0.0
#         valid_number_cats = 0
#         log_frac_pq_per_cat = {}
#         for current_cat in all_cats:
#             numerator_per_cat = 0.0
#             denominator_per_cat = 0.0

#             for b in range(batch_size):
#                 # See if current class exists in the image
#                 if current_cat not in category_indices_batch[b]:
#                     continue 
                
#                 indices = category_indices_batch[b][current_cat]
#                 current_gt_non_zero_mask = gt_non_zero_mask[b, indices]
                
#                 current_foreground_prob = foreground_prob[b, indices] 

#                 # For a TP:
#                 # 1. IOU > 0.5 (true_probability)
#                 # 2. Ground-truth mask should be > 0. (Since IOU function has epsilon factor, it can give 1 IOU for p = 1, g = 0 with eps = 1)
#                 # 3. It should be a foreground object as told by 'foreground_prob' 
#                 tp_indicator = true_probability[b, indices] * current_foreground_prob * current_gt_non_zero_mask
#                 numerator_per_cat += (tp_indicator * ious[b, indices]).sum()
#                 soft_num_tp = tp_indicator.sum()

#                 # For a FN: The metric checks for all valid (non-zero) GT masks against the predictions.
#                 soft_num_fn = (false_probability[b, indices] * current_gt_non_zero_mask).sum()

#                 # For a FP: The influence of a FP can be decreased if the network predicts non-foreground 
#                 # through foreground_prob ~ 0.
#                 soft_num_fp = ((1.0 - current_gt_non_zero_mask) * false_probability[b, indices] * current_foreground_prob).sum()
#                 denominator_per_cat += soft_num_tp + 0.5 * soft_num_fn + 0.5 * soft_num_fp

#             # Add the per-class IoU loss to full loss.
#             if denominator_per_cat > 0:
#                 pq_per_cat = (numerator_per_cat + self.eps) / (denominator_per_cat + self.eps)
#                 # print(f"{current_cat}: pq: {pq_per_cat.item():.3f}, num: {numerator_per_cat.item():.3f}, den: {denominator_per_cat.item():.3f}")
#                 full_pq = full_pq + pq_per_cat
#                 valid_number_cats += 1
#                 log_frac_pq_per_cat[current_cat] = pq_per_cat.item()

#         # Convert to average.
#         full_pq = full_pq / float(valid_number_cats)
#         for key in log_frac_pq_per_cat:
#             log_frac_pq_per_cat[key] = log_frac_pq_per_cat[key] / (full_pq.item() * valid_number_cats + 1e-6)

#         # print(f"FULL iou: {full_pq.item():.4f}")
#         return 1.0 - full_pq, log_frac_pq_per_cat

# def gradcheck_panoptic_quality_loss_3d():
#     # Set up dummy inputs
#     batch_size = 2
#     num_classes = 2
#     depth, height, width = 4, 8, 8
    
#     pan_pred_batch = torch.rand(batch_size, num_classes, depth, height, width, requires_grad=True)
#     pan_gt_batch = torch.rand(batch_size, num_classes, depth, height, width)
#     weights = torch.rand(batch_size, depth, height, width)
#     foreground_prob = torch.rand(batch_size, num_classes)
    
#     category_indices_batch = [
#         {0: [0], 1: [1]},
#         {0: [0], 1: [1]}
#     ]

#     # Initialize the loss function
#     loss_fn = PanopticQualityLoss3D(thing_ids=[0, 1])

#     # Define the test function
#     def test_function(input_tensor):
#         loss, _ = loss_fn(input_tensor, pan_gt_batch, category_indices_batch, weights, foreground_prob, iou_batch_3d)
#         return loss

#     # Perform gradient check
#     torch.set_printoptions(precision=8)
#     result = autograd.gradcheck(test_function, pan_pred_batch, eps=1e-3, atol=1e-3)
    
#     print("Gradient check passed:", result)