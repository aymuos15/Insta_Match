import torch

import cupy as cp
from cucim.skimage import measure as cucim_measure

from overlap_metrics.legacy_dice import legacy_dice as dice

def get_connected_components(img):
    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(img_cupy, return_num=True)
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)
    return labeled_img_torch, num_features

def blob_dice(
    network_outputs: torch.Tensor,
    gt_label: torch.Tensor,
):
    
    multi_label, _ = get_connected_components(gt_label)

    multi_label = multi_label.unsqueeze(0)
    network_outputs = network_outputs.unsqueeze(0)
  
    batch_length = multi_label.shape[0]

    element_blob_loss = []
    # loop over elements
    for element in range(batch_length):
        if element < batch_length:
            end_index = element + 1
        elif element == batch_length:
            end_index = None

        element_label = multi_label[element:end_index, ...]


        element_output = network_outputs[element:end_index, ...]

        # loop through labels
        unique_labels = torch.unique(element_label)

        label_loss = []
        for ula in unique_labels:
            if ula == 0:
                pass
            else:
                # first we need one hot labels
                label_mask = element_label > 0
                # we flip labels
                label_mask = ~label_mask

                # we set the mask to true where our label of interest is located
                label_mask[element_label == ula] = 1

                the_label = element_label == ula

                masked_output = element_output * label_mask

                blob_metric = dice(masked_output, the_label)

                label_loss.append(blob_metric)

        # compute mean
        element_blob_loss.append(torch.mean(torch.stack(label_loss)))

    return torch.mean(torch.stack(element_blob_loss))