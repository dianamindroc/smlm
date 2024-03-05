#from model_architectures.chamfer_distances import ChamferDistanceL2, ChamferDistanceL1
import torch
from model_architectures.chamfer_distance_updated import ChamferDistance
import open3d as o3d

CD = ChamferDistance()

def cd_loss_l1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    #loss = ChamferDistanceL1()
    #dist1, dist2 = loss(pcs1, pcs2)
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0

def l1_cd(pcs1, pcs2):
    #loss = ChamferDistanceL1()
    #dist1, dist2 = loss(pcs1, pcs2)
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2

def l2_cd(pcs1, pcs2):
    #loss = ChamferDistanceL1()
    #dist1, dist2 = loss(pcs1, pcs2)
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return torch.sum(dist1 + dist2)

def cd_loss_l2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    #loss = ChamferDistanceL2()
    #dist1, dist2 = loss(pcs1, pcs2)
    dist1, dist2 = CD(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)

def f_score(pred, gt, th=0.01):
    """
    References: https://github.com/lmb-freiburg/what3d/blob/master/util.py

    Args:
        pred (np.ndarray): (N1, 3)
        gt   (np.ndarray): (N2, 3)
        th   (float): a distance threshhold
    """
    pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

    dist1 = pred.compute_point_cloud_distance(gt)
    dist2 = gt.compute_point_cloud_distance(pred)

    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0