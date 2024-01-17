import numpy as np
import torch
from scipy.stats.qmc import Sobol
import time
import math


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


class Clock:
  def __init__(self):
    self.start_time = None
    self.end_time = None

  def start(self):
    self.start_time = time.time()

  def stop(self):
    self.end_time = time.time()

  def elapsed_time(self):
    return self.end_time - self.start_time

  def reset(self):
    self.start_time = None
    self.end_time = None

  def __str__(self):
    return f"Elapsed time: {self.elapsed_time():.4f}s"


class ReLoBRaLo:
  def __init__(self, alpha:float=0.999, temperature:float=1., rho:float=0.9999, epsilon:float=1e-8):
    self.alpha = alpha
    self.temperature = temperature
    self.rho = rho
    self.epsilon = epsilon


  def __compute_lambda_i_bal(self, L: np.array, i: int, t1_index: int, t2_index: int):
    # L: m * n -> m: number of losses, n: number of iterations
    # i: index of the loss function
    # t1_index: index of the first iteration
    # t2_index: index of the second iteration

    if t1_index < 0 or t2_index < 0: return 0

    m = len(L)
    lambda_i_bal = L[i][t1_index] / (self.temperature * L[i][t2_index] + self.epsilon)
    lambda_i_bal = np.exp(lambda_i_bal)
    _sum = np.sum([np.exp(L[j][t1_index] / (self.temperature * L[j][t2_index] + self.epsilon)) for j in range(m)])
    lambda_i_bal /= _sum
    lambda_i_bal *= m

    return lambda_i_bal


  # def __compute_lambda_i_hist(self, L: np.array, i: int, t_index: int):

  #   if t_index < 0: return 0

  #   lambda_i_hist = self.rho * self.__compute_lambda_i(L, i, t_index-1) + (1 - self.rho) * self.__compute_lambda_i_bal(L, i, t_index, 0)

  #   return lambda_i_hist


  # def __compute_lambda_i(self, L: np.array, i: int, t_index: int):

  #   if t_index < 0: return 0

  #   lambda_i = self.alpha * self.__compute_lambda_i_hist(L, i, t_index) + (1 - self.alpha) * self.__compute_lambda_i_bal(L, i, t_index, t_index-1)

  #   return lambda_i


  # def compute_next_lambdas(self, L: np.array):
  #   # L: m * n -> m: number of losses, n: number of iterations
  #   m = len(L)
  #   n = len(L[0])

  #   next_lambdas = []

  #   for i in range(m):
  #     next_lambdas.append(self.__compute_lambda_i(L, i, n-1))

  #   return next_lambdas


  def __compute_lambda_i_hist(self, L, i, t_index, lambda_i_history):
    if t_index < 0: exit()

    if t_index == 0:
      return 1

    prev_lambda_i = lambda_i_history[i][t_index-1]
    lambda_i_hist = self.rho * prev_lambda_i + (1 - self.rho) * self.__compute_lambda_i_bal(L, i, t_index, 0)

    return lambda_i_hist


  def __compute_lambda_i(self, L, i, t_index, lambda_i_history):
      if t_index < 0: exit()

      lambda_i_hist = self.__compute_lambda_i_hist(L, i, t_index, lambda_i_history)
      lambda_i_bal = self.__compute_lambda_i_bal(L, i, t_index, t_index-1)
      lambda_i = self.alpha * lambda_i_hist + (1 - self.alpha) * lambda_i_bal

      return lambda_i


  def compute_next_lambdas(self, L):
      m = len(L)
      n = len(L[0])

      lambda_i_history = [[0 for _ in range(n)] for _ in range(m)]
      next_lambdas = []

      for t in range(n):
          for i in range(m):
              lambda_i_history[i][t] = self.__compute_lambda_i(L, i, t, lambda_i_history)

      for i in range(m):
          next_lambdas.append(lambda_i_history[i][n-1])

      return next_lambdas