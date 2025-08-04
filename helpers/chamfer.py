from model_architectures.chamfer_distance_updated import ChamferDistance
import torch

# Initialize the Chamfer Distance
cd = ChamferDistance()

# Helper function to quickly get Chamfer Distance between 2 point clouds
def compare_pc(pc1, pc2, cd_type, ignore_zeros=False):
    """
    Calculate Chamfer distance between two point clouds.
    
    Args:
        pc1: First point cloud (B, N, 3)
        pc2: Second point cloud (B, M, 3)
        cd_type: Type of distance metric ('L1' or 'L2')
        ignore_zeros: For compatibility, not used in this implementation
    
    Returns:
        Chamfer distance between the point clouds
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pc1 = pc1.to(device).float()
    pc2 = pc2.to(device).float()
    
    # The updated implementation returns (dist1, dist2, idx1, idx2)
    dist1, dist2, _, _ = cd(pc1, pc2)
    
    if cd_type.upper() == 'L1':
        # L1 Chamfer distance
        cd_dist = (torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))) / 2.0
    elif cd_type.upper() == 'L2':
        # L2 Chamfer distance (squared)
        cd_dist = (torch.mean(dist1) + torch.mean(dist2)) / 2.0
    else:
        raise NotImplementedError(f"Distance type '{cd_type}' is not implemented. Use 'L1' or 'L2'.")
    
    return cd_dist

