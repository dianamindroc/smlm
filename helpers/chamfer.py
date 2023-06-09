from model_architectures.chamfer_distances import ChamferDistanceL2, ChamferDistanceL1

import torch

cd_l2 = ChamferDistanceL2()
cd_l1 = ChamferDistanceL1()


def compare_pc(pc1, pc2, cd_type, ignore_zeros=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if cd_type == 'L1':
        cd = ChamferDistanceL1()
    elif cd_type == 'L2':
        cd = ChamferDistanceL2(ignore_zeros=ignore_zeros)
    else:
        raise NotImplementedError("This distance is not implemented or does not exist.")
    difference = cd(pc1.to(device).float(), pc2.to(device).float())
    return difference

