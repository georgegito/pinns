import numpy as np
import torch

def stack_xyz(x, y, z):
  return np.stack((x, y, z), axis=-1)

def stack_xyzt(x, y, z, t):
  return np.stack((x, y, z, t), axis=-1)

def xyz_for_all_t(xyz, t):
  return np.array([np.concatenate([_xyz, [_t]]) for _xyz in xyz for _t in t])

def tensor_from_array(arr, device):
  return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=True)

# x = [1, 2, 3, 4]
# y = [4, 5, 6, 7]
# z = [7, 8, 9, 10]

# xyz = stack_xyz(x, y, z)

# print(xyz.shape)


# t = [1, 2]

# xyzt = xyz_for_all_t(xyz, t)

# tensor_xyzt = tensor_from_array(xyzt, 'cpu')

# # print(tensorxyzt.shape)

# # print(torch.tensor(xyz).shape)

# out_filepath = "/Users/ggito/repos/pinns/data/"
# pinn = torch.load(out_filepath + 'pinn8.pt').to('cpu')
# # pinn.train()
# pinn.eval()

# print(pinn(tensor_xyzt))

# [casc, dasasd, asdasda, daada] = pinn(tensor_xyzt).


