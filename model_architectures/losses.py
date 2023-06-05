from model_architectures.chamfer_distances import ChamferDistanceL2, ChamferDistanceL1
import torch

def cd_loss_l1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    loss = ChamferDistanceL1()
    dist1, dist2 = loss(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_l2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    loss = ChamferDistanceL2()
    dist1, dist2 = loss(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)
