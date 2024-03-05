import importlib
import os

import torch
from torch import nn
from torch.autograd import Function


chamfer_found = importlib.find_loader("chamfer_3D") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer 3D")

    from torch.utils.cpp_extension import load
    chamfer_3D = load(name="chamfer_3D",
          sources=[
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
              ])
    # print("Loaded JIT 3D CUDA chamfer distance")

else:
    import chamfer_3D
    # print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        """
        xyz1: (B, N, 3)
        xyz2: (B, M, 3)
        """
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2

class chamfer_3DFunction_single(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        """
        xyz1: List of tensors of shape (N, 3)
        xyz2: List of tensors of shape (M, 3)
        """
        print('Shape of xyz1 in forward is ', xyz1.shape)
        batchsize = len(xyz1)
        device = xyz1[0].device

        dist1_list = []
        dist2_list = []
        idx1_list = []
        idx2_list = []

        for i in range(batchsize):
            n = xyz1[i].size(0)
            m = xyz2[i].size(0)

            dist1 = torch.zeros(n)
            dist2 = torch.zeros(m)
            idx1 = torch.zeros(n, dtype=torch.int)
            idx2 = torch.zeros(m, dtype=torch.int)

            dist1 = dist1.to(device)
            dist2 = dist2.to(device)
            idx1 = idx1.to(device)
            idx2 = idx2.to(device)

            chamfer_3D.forward(xyz1[i], xyz2[i], dist1, dist2, idx1, idx2)

            dist1_list.append(dist1)
            dist2_list.append(dist2)
            idx1_list.append(idx1)
            idx2_list.append(idx2)

        ctx.save_for_backward(*dist1_list, *dist2_list, *idx1_list, *idx2_list)
        return torch.stack(dist1_list), torch.stack(dist2_list), torch.stack(idx1_list), torch.stack(idx2_list)

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        dist1_list, dist2_list, idx1_list, idx2_list = ctx.saved_tensors
        batchsize = len(dist1_list)
        device = dist1_list[0].device

        gradxyz1_list = []
        gradxyz2_list = []

        for i in range(batchsize):

            if len(dist1_list[i].size()) == 0:  # Check if the tensor has dimensions
                continue  # Skip if the tensor has no dimensions

            n = dist1_list[i].size(0)
            m = dist2_list[i].size(0)

            gradxyz1 = torch.zeros(n, 3)
            gradxyz2 = torch.zeros(m, 3)

            gradxyz1 = gradxyz1.to(device)
            gradxyz2 = gradxyz2.to(device)

            chamfer_3D.backward(
                gradxyz1, gradxyz2, graddist1[i], graddist2[i], gradidx1[i], gradidx2[i]
            )

            gradxyz1_list.append(gradxyz1)
            gradxyz2_list.append(gradxyz2)

        return gradxyz1_list, gradxyz2_list

class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, input1, input2):
       """
       input1: (B, N, 3)
       input2: (B, M, 3)
       """
       dist1, dist2, _, _ = chamfer_3DFunction.apply(input1, input2)
       return dist1, dist2

#     def forward(self, input1, input2):
#         """
#         Adjusted to handle lists of tensors of shape (N, 3) and (M, 3), where N and M can vary.
#         """
#         dist1_total = 0
#         dist2_total = 0
#         for pc1, pc2 in zip(input1, input2):
#             pc1 = pc1.unsqueeze(0)
#             pc2 = pc2.unsqueeze(0)
#             print('Shape of pc1 is ', pc1.shape)
#             print('Shape of pc2 is ', pc2.shape)
#             dist1, dist2, _, _ = chamfer_3DFunction_single.apply(pc1, pc2)
#             dist1_total += dist1.mean()
#             dist2_total += dist2.mean()
#         return dist1_total / len(input1), dist2_total / len(input1)