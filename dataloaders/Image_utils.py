import numpy as np
import torch
from collections import OrderedDict


def decode_segmap_sequence(label_masks, label_colours = OrderedDict([(0, 0), (1, 255), (2, 191), (3, 128)])):
    masks = [decode_segmap(label_mask) for label_mask in label_masks]
    masks = torch.from_numpy(np.array(masks))
    return masks


def decode_segmap(label_mask, label_colours = OrderedDict([(0, 0), (1, 255), (2, 191), (3, 128)]), plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    gray_mask = label_mask.copy()
    for target_pixel in label_colours:
        np.place(gray_mask, gray_mask==target_pixel, label_colours[target_pixel])
    
    if plot:
        plt.imshow(gray_mask, cmap="gray")
        plt.show()
    else:
        return gray_mask
