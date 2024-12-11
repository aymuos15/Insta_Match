import torch 

def legacy_dice(im1, im2):

    '''
    Generic Dice Score
    '''

    """
    im1: torch.Tensor of shape (H, W, D) or (H, W)
    im2: torch.Tensor of shape (H, W, D) or (H, W)
    """

    intersection = torch.sum(im1 * im2)
    sum_im1 = torch.sum(im1)
    sum_im2 = torch.sum(im2)
    return 2.0 * intersection / (sum_im1 + sum_im2)