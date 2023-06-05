import torch
import numpy as np

class ToTensor(object):
    """
    Transforms the point cloud to tensor
    """
    def __call__(self, sample):
        if sample is not None:
            pc, label, path = sample['pc'], sample['label'], sample['path']
            pc = torch.tensor(pc, dtype=torch.float)
            return {
                'pc': pc,
                'label': label,
                'path': path
            }


class Padding(object):
    """
    Pads the point cloud to the highest founds size in the dataset
    """
    def __init__(self, highest_shape):
        self.highest_shape = highest_shape

    def __call__(self, sample):
        pc = sample['pc']
        target = (self.highest_shape, pc.shape[-1])
        padding = [(0, target[0] - pc.shape[0]), (0, 0)]
        if any(num < 0 for nums in padding for num in nums):
            return sample
        padded_arr = np.pad(pc, padding)
        sample['pc'] = padded_arr
        return sample
