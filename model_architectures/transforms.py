import torch
import numpy as np

class ToTensor(object):
    """
    Transforms the point cloud to tensor
    """
    def __call__(self, sample):
        # if sample is not None:
        #     pc, label, path = sample['pc'], sample['label'], sample['path']
        #     pc = torch.tensor(pc, dtype=torch.float)
        #     return {
        #         'pc': pc,
        #         'label': label,
        #         'path': path
        #     }
        if sample is not None:
            sample = torch.tensor(sample, dtype=torch.float)
        return sample

class Padding(object):
    """
    Pads the point cloud to the highest found size in the dataset
    """
    def __init__(self, highest_shape):
        self.highest_shape = highest_shape

    def __call__(self, sample):
        # Assuming sample is a 2D array with shape (num_points, 3)
        target = (self.highest_shape, sample.shape[-1])  # sample.shape[-1] should be 3 for x, y, z coordinates
        padding = [(0, target[0] - sample.shape[0]), (0, 0)]
        if any(num < 0 for nums in padding for num in nums):
            return sample  # Return the original sample if no padding is needed
        padded_arr = np.pad(sample, padding, mode='constant', constant_values=0)  # Pad with zeros
        mask = np.zeros((self.highest_shape, sample.shape[-1]))
        mask[:len(sample), :] = 1
        return padded_arr, mask
