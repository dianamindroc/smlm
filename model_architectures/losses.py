#from model_architectures.chamfer_distances import ChamferDistanceL2, ChamferDistanceL1
import torch
from model_architectures.chamfer_distance_updated import ChamferDistance
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.optimize import minimize


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
    dist1, dist2, idx1, idx2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def l1_cd(pcs1, pcs2):
    #loss = ChamferDistanceL1()
    #dist1, dist2 = loss(pcs1, pcs2)
    dist1, dist2, _, _ = CD(pcs1, pcs2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2


def l2_cd(pcs1, pcs2):
    #loss = ChamferDistanceL1()
    #dist1, dist2 = loss(pcs1, pcs2)
    dist1, dist2, _, _ = CD(pcs1, pcs2)
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


def mlp_loss_function(label, out_mlp):
    criterion = torch.nn.CrossEntropyLoss()
    # Ensure labels are in the correct shape and type
    targets = label.long()
    mean_loss = criterion(out_mlp, targets)
    return mean_loss


def calc_cd(output, gt, calc_f1=False, return_raw=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = CD(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0), dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, 0.0001)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res


def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    # from paper
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res


def _compute_density(points, k=10):
    """
    Helper function. Compute the density of points in a point cloud using k-nearest neighbors.

    Args:
    - points (torch.Tensor): The point cloud [B, N, 3].
    - k (int): Number of nearest neighbors to consider.

    Returns:
    - densities (torch.Tensor): Estimated density for each point [B, N].
    """
    B, N, _ = points.shape
    densities = torch.zeros(B, N)

    for b in range(B):
        point_cloud = points[b].detach().cpu().numpy()
        tree = cKDTree(point_cloud)
        distances, _ = tree.query(point_cloud, k=k + 1)  # k+1 because the point itself is included
        avg_distances = torch.tensor(distances[:, 1:].mean(axis=1), dtype=torch.float32)  # Exclude the first distance
        densities[b] = 1 / (avg_distances + 1e-6)  # Add a small value to avoid division by zero

    return densities


def _with_density(pred, gt, density_pred, density_gt):
    # pred: predicted point cloud [B, N, 3]
    # gt: ground truth point cloud [B, M, 3]
    # density_pred: density of predicted points [B, N]
    # density_gt: density of ground truth points [B, M]
    B, N, _ = pred.shape
    _, M, _ = gt.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Compute pairwise distances
    pred_expand = pred.unsqueeze(2).expand(B, N, M, 3)
    gt_expand = gt.unsqueeze(1).expand(B, N, M, 3)
    dist = torch.norm(pred_expand - gt_expand, dim=3)

    # Weighted Chamfer Distance
    density_pred_expand = density_pred.unsqueeze(2).expand(B, N, M)
    density_gt_expand = density_gt.unsqueeze(1).expand(B, N, M)
    density_pred_expand, density_gt_expand = density_pred_expand.to(device), density_gt_expand.to(device)
    # For each predicted point, find the minimum distance to any ground truth point, weighted by density
    loss_pred = torch.min(dist * density_pred_expand, dim=2)[0]  # Shape: [B, N]
    # For each ground truth point, find the minimum distance to any predicted point, weighted by density
    loss_gt = torch.min(dist * density_gt_expand, dim=1)[0]  # Shape: [B, M]

    return loss_pred.mean() + loss_gt.mean()


def _sparsity_regularization(points, lambda_reg):
    # points: point cloud [B, N, 3]
    B, N, _ = points.shape
    return lambda_reg * N


def safe_normalize(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0 or torch.isnan(std):
        return tensor - mean  # If std is zero, normalization would lead to NaN
    else:
        return (tensor - mean) / std


#### Try it own version of density-aware

def gaussian_kernel(dist, bandwidth):
    return torch.exp(-0.5 * (dist / bandwidth) ** 2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))


def kernel_density_estimate(points, bandwidth):
    batch_size, num_points, _ = points.size()
    points_expand = points.unsqueeze(2).expand(batch_size, num_points, num_points, 3)
    dist = torch.norm(points_expand - points_expand.transpose(1, 2), dim=3)
    kde = gaussian_kernel(dist, bandwidth)
    density_estimates = kde.sum(dim=2) / (num_points - 1)
    return density_estimates


def log_likelihood(points, bandwidth):
    density_estimates = compute_density(points, bandwidth)
    log_likelihoods = torch.log(density_estimates).sum(dim=1)
    return log_likelihoods


def optimize_bandwidth(points):
    def objective(h):
        h = torch.tensor(h, dtype=torch.float32)
        return -log_likelihood(points, h).mean().item()

    result = minimize(objective, x0=0.1, bounds=[(1e-3, 1.0)])
    return result.x


def compute_density(points, bandwidth, chunk_size=100):
    batch_size, num_points, _ = points.size()
    densities = torch.zeros(batch_size, num_points).to(points.device)
    bandwidth = bandwidth.to(points.device)

    for i in range(0, num_points, chunk_size):
        end_i = min(i + chunk_size, num_points)
        for j in range(0, num_points, chunk_size):
            end_j = min(j + chunk_size, num_points)
            points_chunk = points[:, i:end_i, :].unsqueeze(2)  # [batch_size, chunk_size, 1, 3]
            other_points_chunk = points[:, j:end_j, :].unsqueeze(1)  # [batch_size, 1, chunk_size, 3]
            dist_chunk = torch.norm(points_chunk - other_points_chunk, dim=3)  # [batch_size, chunk_size, chunk_size]
            kde_chunk = torch.exp(-0.5 * (dist_chunk / bandwidth) ** 2)
            densities[:, i:end_i] += kde_chunk.sum(dim=2)

    densities /= (num_points - 1)
    return densities


class DensityAwareChamferLoss(nn.Module):
    def __init__(self):
        super(DensityAwareChamferLoss, self).__init__()
        self.bandwidth_pred = None
        self.bandwidth_gt = None

    def forward(self, predicted, ground_truth):
        # Compute Chamfer distances
        dist_pred_to_gt, dist_gt_to_pred, _, _ = CD(predicted, ground_truth)

        # Compute or update bandwidth using MLE every few epochs
        if self.bandwidth_pred is None or self.bandwidth_gt is None:
            self.bandwidth_pred = optimize_bandwidth(predicted)
            self.bandwidth_gt = optimize_bandwidth(ground_truth)

        bandwidth_pred = torch.tensor(self.bandwidth_pred, dtype=torch.float32)
        bandwidth_gt = torch.tensor(self.bandwidth_gt, dtype=torch.float32)

        # Compute densities
        density_pred = compute_density(predicted, bandwidth_pred)
        density_gt = compute_density(ground_truth, bandwidth_gt)

        # Weight distances by densities
        weighted_dist_pred_to_gt = dist_pred_to_gt / (density_pred + 1e-6)
        weighted_dist_gt_to_pred = dist_gt_to_pred / (density_gt + 1e-6)

        # Compute mean weighted distances
        loss_pred_to_gt = weighted_dist_pred_to_gt.mean()
        loss_gt_to_pred = weighted_dist_gt_to_pred.mean()

        # Combine losses
        total_loss = loss_pred_to_gt + loss_gt_to_pred

        return total_loss

#TODO: clean up duplicate or useless things