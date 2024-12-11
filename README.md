# Insta-Match: A Library to perform Instance Based Evaluation and Training for Semantic Segmentation Tasks, On the GPU. 

Like [panoptica](https://github.com/BrainLesion/panoptica/tree/main), this library is built for Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps. Our key points:

- *Metrics are integrated for utilisation on the GPU. 
- Evluation pipelines for holistic instance-wise evaluation.
- Loss Function implementations as well.
- Easy integration helper functions to improve instance computations such as a GPU based connected components wrapper for Torch.

### Matching Schemes:

- One-to-One: Panoptic Quality (Dice) | [Kirillov et al.](https://arxiv.org/abs/1801.00868)
- Partial-to-One: Region Dice | [Jaus et al.](https://arxiv.org/abs/2410.18684)
- Partial-to-One: Blob Dice | [Kofler et al.](https://arxiv.org/abs/2205.08209)
- Many-to-One: Lesion-wise Dice | [BraTS-Mets Group](https://github.com/rachitsaluja/BraTS-2023-Metrics)
- Many-to-Many: Cluster Dice | [Kundu et al.]()

### Losses

All the above metrics have their loss functions presented as well. Additionally:

- Blob Loss | [Kofler et al.](https://arxiv.org/abs/2205.08209)
- Region Cross Entropy | [Liu et al.](https://arxiv.org/abs/2104.08717)

### Explanation

For a detailed analysis and understanding of the matching schemes and losses working -- `analysis` directory

- `losscheck.ipynb` - Checks for gradients in the loss functions through gradcheck.

- `viz.ipynb` - Is an attempt to visualise the matching schemes.

### TODOs

- [ ] Need to add [modified panoptic quality](https://lightning.ai/docs/torchmetrics/stable/detection/modified_panoptic_quality.html)

- [ ] Need to adjust everything to multi-class settings

- [ ] Need more verification for distance metrics

- [ ] *Need to get distance metrics on the GPU.

- [ ] Add More Instance Losses such as 

    - Universal Loss Reweighting | [Shirokikh et al.](https://arxiv.org/abs/2007.10033)
    - Instance Loss Functions (built on top of blob loss) | [Rachmadi et al.](https://www.sciencedirect.com/science/article/pii/S0010482524004980)

- [ ] Need to make everything in 2D as well.