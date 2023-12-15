import numpy as np
import torch

# def stack_xyz(x, y, z):
#   return np.stack((x, y, z), axis=-1)

def stack_xyzt_tensors(x, y, z, t):
  return torch.stack((x, y, z, t), axis=-1)

def stack_xyz_tensors(x, y, z):
  return torch.stack((x, y, z), axis=-1)

# def xyz_for_all_t(xyz, t):
#   return np.array([np.concatenate([_xyz, [_t]]) for _xyz in xyz for _t in t])

def tensor_from_array(arr, device, requires_grad):
  return torch.tensor(arr, device=device, requires_grad=requires_grad, dtype=torch.float32)

# x = tensor_from_array([1, 2, 3], 'cpu')
# y = tensor_from_array([4, 5, 6], 'cpu')
# z = tensor_from_array([7, 8, 9], 'cpu')
# t = tensor_from_array([10, 11, 12], 'cpu')

# # xyz = stack_xyz(x, y, z)
# tensor_xyzt = stack_xyzt(x, y, z, t)

# # print(xyz.shape)
# # print(xyzt)
# print(tensor_xyzt.shape)


# # t = [1, 2]

# # xyzt = xyz_for_all_t(xyz, t)

# # tensor_xyzt = tensor_from_array(xyzt, 'cpu')

# # # print(tensorxyzt.shape)

# # # print(torch.tensor(xyz).shape)

# out_filepath = "/Users/ggito/repos/pinns/data/"
# pinn = torch.load(out_filepath + 'pinn8.pt').to('cpu')
# # pinn.train()
# pinn.eval()

# print(pinn(tensor_xyzt))

# # [casc, dasasd, asdasda, daada] = pinn(tensor_xyzt).


