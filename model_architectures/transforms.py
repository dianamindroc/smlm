import torch
import numpy as np

class ToTensor(object):
    """
    Converts point clouds and masks in the sample to PyTorch tensors.
    """
    def __call__(self, sample):
        # Convert point clouds and their masks to tensors
        if 'pc' in sample:
            sample['pc'] = torch.tensor(sample['pc'], dtype=torch.float)
        if 'partial_pc' in sample:
            sample['partial_pc'] = torch.tensor(sample['partial_pc'], dtype=torch.float)
        if 'pc_mask' in sample:
            sample['pc_mask'] = torch.tensor(sample['pc_mask'], dtype=torch.float)
        if 'partial_pc_mask' in sample:
            sample['partial_pc_mask'] = torch.tensor(sample['partial_pc_mask'], dtype=torch.float)

        # Convert 'label' to tensor if it's numeric
        if 'label' in sample and isinstance(sample['label'], (list, np.ndarray)):
            sample['label'] = torch.tensor(sample['label'], dtype=torch.float)
        elif 'label' in sample and isinstance(sample['label'], (int, float)):
            sample['label'] = torch.tensor([sample['label']], dtype=torch.float)

        # Non-numeric data like 'path' are left as is

        return sample

class Padding(object):
    """
    Pads the point clouds 'pc' and 'partial_pc' in the sample to the highest found size in the dataset.
    """
    def __init__(self, highest_shape):
        self.highest_shape = highest_shape

    def pad_point_cloud(self, point_cloud):
        """
        Pads a single point cloud array to the specified highest shape.
        """
#         target_shape = (self.highest_shape, point_cloud.shape[-1])  # Assuming point cloud shape is (num_points, num_dimensions)
#         padding = [(0, target_shape[0] - point_cloud.shape[0]), (0, 0)]  # Calculate padding for points and dimensions
#         if any(num < 0 for nums in padding for num in nums):
#             return point_cloud, np.ones(point_cloud.shape)  # Return original point cloud and a mask of ones if no padding is needed
#         padded_point_cloud = np.pad(point_cloud, padding, mode='constant', constant_values=0)  # Pad with zeros
#         mask = np.zeros(target_shape)
#         mask[:point_cloud.shape[0], :] = 1  # Indicate original points with ones in the mask
#         return padded_point_cloud, mask

        num_points_needed = self.highest_shape - point_cloud.shape[0]
        if num_points_needed > 0:
            # Select random indices from the point cloud to duplicate
            indices_to_duplicate = np.random.choice(point_cloud.shape[0], num_points_needed, replace=True)
            duplicated_points = point_cloud[indices_to_duplicate]

            # Concatenate the original point cloud with the duplicated points
            padded_point_cloud = np.vstack((point_cloud, duplicated_points))
        else:
            padded_point_cloud = point_cloud

        # Create a mask indicating original points with ones and duplicated points with zeros
        mask = np.ones(self.highest_shape)
        mask[:point_cloud.shape[0]] = 1  # Original points
        return padded_point_cloud, mask

    def __call__(self, sample):
        """
        Pads 'pc' and 'partial_pc' in the sample and adds padding masks.
        """
        pc, pc_mask = self.pad_point_cloud(sample['pc'])
        partial_pc, partial_pc_mask = self.pad_point_cloud(sample['partial_pc'])

        # Update the sample dictionary with padded point clouds and their masks
        sample['pc'] = pc
        sample['partial_pc'] = partial_pc
        sample['pc_mask'] = pc_mask
        sample['partial_pc_mask'] = partial_pc_mask

        return sample