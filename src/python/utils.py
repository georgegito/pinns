import numpy as np
import torch
from scipy.stats.qmc import Sobol

def stack_xyzt_tensors(x: torch.tensor, y: torch.tensor, z: torch.tensor, t: torch.tensor) -> torch.tensor:
  return torch.stack((x, y, z, t), axis=-1)

def stack_xyz_tensors(x: torch.tensor, y: torch.tensor, z: torch.tensor) -> torch.tensor:
  return torch.stack((x, y, z), axis=-1)

def tensor_from_array(arr: np.ndarray, device: torch.device, requires_grad: bool) -> torch.tensor:
  return torch.tensor(arr, device=device, requires_grad=requires_grad, dtype=torch.float32)

def sample_points_in_domain(_min: float, _max: float, num_samples: int) -> np.ndarray:
  return np.random.uniform(_min, _max, size=num_samples)

def sample_points_in_domain_3d(_min: float, _max: float, num_samples: int) -> np.ndarray:
  samples = np.random.uniform(_min, _max, size=(num_samples, 3))

  x = samples[:, 0]
  y = samples[:, 1]
  z = samples[:, 2]

  return x, y, z

def sample_points_in_domain_3d(_min: float, _max: float, num_samples: int) -> np.ndarray:
  samples = np.random.uniform(_min, _max, size=(num_samples, 3))

  x = samples[:, 0]
  y = samples[:, 1]
  z = samples[:, 2]

  return x, y, z

def qmc_sample_points_in_domain_3d(_min: float, _max: float, num_samples: int) -> np.ndarray:
  sobol = Sobol(d=3)  # 3 dimensions
  samples = sobol.random_base2(m=int(np.log2(num_samples)))

  samples = _min + (_max - _min) * samples

  x = samples[:, 0]
  y = samples[:, 1]
  z = samples[:, 2]
  
  return x, y, z

def qmc_sample_points_in_domain_3d_space_time(_min: float, _max: float, _t_min: float, _t_max: float, num_samples: int) -> np.ndarray:
  sobol = Sobol(d=3)  # 3 dimensions (2 spatial + 1 temporal)
  samples = sobol.random_base2(m=int(np.log2(num_samples)))

  # Apply domain transformations for space
  for i in range(2):  # For x, y
      samples[:, i] = _min + (_max - _min) * samples[:, i]

  # Apply domain transformation for time
  samples[:, 2] = _t_min + (_t_max - _t_min) * samples[:, 2]

  x = samples[:, 0]
  y = samples[:, 1]
  t = samples[:, 2]
  
  return x, y, z, t

def qmc_sample_points_in_domain_4d_space_time(_min: float, _max: float, _t_min: float, _t_max: float, num_samples: int) -> np.ndarray:
  sobol = Sobol(d=4)  # 4 dimensions (3 spatial + 1 temporal)
  samples = sobol.random_base2(m=int(np.log2(num_samples)))

  # Apply domain transformations for space
  for i in range(3):  # For x, y, z
      samples[:, i] = _min + (_max - _min) * samples[:, i]

  # Apply domain transformation for time
  samples[:, 3] = _t_min + (_t_max - _t_min) * samples[:, 3]

  x = samples[:, 0]
  y = samples[:, 1]
  z = samples[:, 2]
  t = samples[:, 3]
  
  return x, y, z, t

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