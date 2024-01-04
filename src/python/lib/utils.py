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


def sample_points_in_domain_3d(_x_min: float, _x_max: float, 
                               _y_min: float, _y_max: float, 
                               _z_min: float, _z_max: float, 
                               num_samples: int) -> np.ndarray:
  samples = np.random.uniform(0, 1, size=(num_samples, 3))

  # Apply domain transformations for space
  samples[:, 0] = _x_min + (_x_max - _x_min) * samples[:, 0]
  samples[:, 1] = _y_min + (_y_max - _y_min) * samples[:, 1]
  samples[:, 2] = _z_min + (_z_max - _z_min) * samples[:, 2]

  x = samples[:, 0]
  y = samples[:, 1]
  z = samples[:, 2]

  return x, y, z

def sample_points_in_domain_4d_space_time(_x_min: float, _x_max: float, 
                                          _y_min: float, _y_max: float, 
                                          _z_min: float, _z_max: float, 
                                          _t_min: float, _t_max: float, 
                                          num_samples: int) -> np.ndarray:
  # Generate samples for the spatial dimensions
  samples = np.random.uniform(0, 1, size=(num_samples, 4))

  # Apply domain transformations for space
  samples[:, 0] = _x_min + (_x_max - _x_min) * samples[:, 0]
  samples[:, 1] = _y_min + (_y_max - _y_min) * samples[:, 1]
  samples[:, 2] = _z_min + (_z_max - _z_min) * samples[:, 2]

  # Apply domain transformations for space
  samples[:, 3] = _t_min + (_t_max - _t_min) * samples[:, 3]

  x = samples[:, 0]
  y = samples[:, 1]
  z = samples[:, 2]
  t = samples[:, 3]

  return x, y, z, t

def qmc_sample_points_in_domain_2d(_x_min: float, _x_max: float, 
                                   _y_min: float, _y_max: float, 
                                   num_samples: int) -> np.ndarray:
  sobol = Sobol(d=2)  # 2 dimensions
  samples = sobol.random_base2(m=int(np.log2(num_samples)))

  # Apply domain transformations for space
  samples[:, 0] = _x_min + (_x_max - _x_min) * samples[:, 0]
  samples[:, 1] = _y_min + (_y_max - _y_min) * samples[:, 1]

  x = samples[:, 0]
  y = samples[:, 1]
  
  return x, y

def qmc_sample_points_in_domain_3d(_x_min: float, _x_max: float, 
                                   _y_min: float, _y_max: float, 
                                   _z_min: float, _z_max: float, 
                                   num_samples: int) -> np.ndarray:
  sobol = Sobol(d=3)  # 3 dimensions
  samples = sobol.random_base2(m=int(np.log2(num_samples)))

  # Apply domain transformations for space
  samples[:, 0] = _x_min + (_x_max - _x_min) * samples[:, 0]
  samples[:, 1] = _y_min + (_y_max - _y_min) * samples[:, 1]
  samples[:, 2] = _z_min + (_z_max - _z_min) * samples[:, 2]

  x = samples[:, 0]
  y = samples[:, 1]
  z = samples[:, 2]
  
  return x, y, z


def qmc_sample_points_in_domain_3d_space_time(_x_min: float, _x_max: float, 
                                              _y_min: float, _y_max: float, 
                                              _t_min: float, _t_max: float, 
                                              num_samples: int) -> np.ndarray:
  sobol = Sobol(d=3)  # 3 dimensions (2 spatial + 1 temporal)
  samples = sobol.random_base2(m=int(np.log2(num_samples)))

  # Apply domain transformations for space
  samples[:, 0] = _x_min + (_x_max - _x_min) * samples[:, 0]
  samples[:, 1] = _y_min + (_y_max - _y_min) * samples[:, 1]

  # Apply domain transformation for time
  samples[:, 2] = _t_min + (_t_max - _t_min) * samples[:, 2]

  x = samples[:, 0]
  y = samples[:, 1]
  t = samples[:, 2]
  
  return x, y, t


def qmc_sample_points_in_domain_4d_space_time(_x_min: float, _x_max: float,
                                              _y_min: float, _y_max: float, 
                                              _z_min: float, _z_max: float, 
                                              _t_min: float, _t_max: float, 
                                              num_samples: int) -> np.ndarray:
  sobol = Sobol(d=4)  # 4 dimensions (3 spatial + 1 temporal)
  samples = sobol.random_base2(m=int(np.log2(num_samples)))

  # Apply domain transformations for space
  samples[:, 0] = _x_min + (_x_max - _x_min) * samples[:, 0]
  samples[:, 1] = _y_min + (_y_max - _y_min) * samples[:, 1]
  samples[:, 2] = _z_min + (_z_max - _z_min) * samples[:, 2]

  # Apply domain transformation for time
  samples[:, 3] = _t_min + (_t_max - _t_min) * samples[:, 3]

  x = samples[:, 0]
  y = samples[:, 1]
  z = samples[:, 2]
  t = samples[:, 3]

  return x, y, z, t


def calculate_statistics(*args: np.ndarray):
  # Combine the input arrays into a single array
  samples = np.stack(args, axis=-1)

  means = np.mean(samples, axis=0)
  variances = np.var(samples, axis=0)
  
  # Calculating correlation matrix
  correlation_matrix = np.corrcoef(samples.T)  # Transpose for correct dimensionality

  return means, variances, correlation_matrix


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


def nearest_power_of_2(n):
    return 2 ** round(np.log2(n))


def get_device() -> torch.device:
  if torch.backends.mps.is_available():
    return torch.device("mps")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")