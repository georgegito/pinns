import numpy as np
import torch

def stack_xyzt_tensors(x: torch.tensor, y: torch.tensor, z: torch.tensor, t: torch.tensor) -> torch.tensor:
  return torch.stack((x, y, z, t), axis=-1)

def stack_xyz_tensors(x: torch.tensor, y: torch.tensor, z: torch.tensor) -> torch.tensor:
  return torch.stack((x, y, z), axis=-1)

def tensor_from_array(arr: np.ndarray, device: torch.device, requires_grad: bool) -> torch.tensor:
  return torch.tensor(arr, device=device, requires_grad=requires_grad, dtype=torch.float32)

def sample_points_in_domain(_min, _max, num_of_samples) -> np.ndarray:
  return np.random.uniform(_min, _max, size=num_of_samples)

def zeros(num: int) -> np.ndarray:
  return np.zeros(num)

def ones(num: int) -> np.ndarray:
  return np.ones(num)

def grad(x: torch.Tensor, y: torch.Tensor, create_graph=True) -> torch.Tensor:
  return torch.autograd.grad(x, y, grad_outputs=torch.ones_like(x), create_graph=create_graph, retain_graph=True, only_inputs=True)[0]
  
def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> float:
  for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate
  return param_group['lr']