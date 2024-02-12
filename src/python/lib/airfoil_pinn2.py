import torch
import torch.nn as nn
import numpy as np
import utils
import os
import pandas as pd
from matplotlib import pyplot as plt
from naca4digit_airfoil import Naca4DigitAirfoil

class AirfoilPINN(nn.Module):

  def __init__(self, hidden_units: int, airfoil: Naca4DigitAirfoil, model_name: str = None):

    super(AirfoilPINN, self).__init__()

    self.airfoil = airfoil

    self.layers = nn.ModuleList()

    self.input_dim = 2
    self.output_dim = 3

    _in_units = self.input_dim
    for units in hidden_units:
      layer = nn.Linear(_in_units, units)
      nn.init.xavier_normal_(layer.weight)  # Apply Xavier initialization
      self.layers.append(layer)
      _in_units = units

    output_layer = nn.Linear(_in_units, self.output_dim)

    nn.init.xavier_normal_(output_layer.weight)  # Apply Xavier initialization

    self.layers.append(output_layer)

    self.logs = {"total_loss": [], 
                 "pde_ns_loss": [], "pde_ps_loss": [],
                 "bc_in_loss": [], "bc_out_loss": [], 
                 "bc_down_loss": [], "bc_up_loss": [], 
                 "surface_loss": [], "interior_loss": []}

    self.lambdas = {"pde_ns": [], "pde_ps": [], 
                   "bc_in": [], "bc_out": [], 
                   "bc_down": [], "bc_up": [], 
                   "surface": [], "interior": []}

    self.curent_total_loss = -1
    self.current_pde_nv_loss = -1
    self.current_pde_ps_loss = -1
    self.current_bc_in_loss = -1
    self.current_bc_out_loss = -1
    self.current_bc_down_loss = -1
    self.current_bc_up_loss = -1
    self.current_surface_loss = -1
    self.current_interior_loss = -1

    self.epoch = 0
    self.hidden_units = hidden_units

    if model_name is None:
      self.model_name = utils.NameGenerator().generate_name()
    else:
      self.model_name = model_name

    self.lambda_pde_ns = .1
    self.lambda_pde_ps = .1
    self.lambda_bc_in = .1
    self.lambda_bc_out = .1
    self.lambda_bc_down = .1
    self.lambda_bc_up = .1
    self.lambda_surface = .1
    self.lambda_interior = .1


  def forward(self, input: torch.Tensor) -> torch.Tensor:
    for layer in self.layers[:-1]:
      # output = torch.sigmoid(layer(input))
      output = torch.tanh(layer(input))
      # output = torch.relu(layer(input))
      input = output
    output = self.layers[-1](input)
    return output


  def loss(
      self, 
      x_f: torch.Tensor,
      y_f: torch.Tensor,
      input_b_in: torch.Tensor,
      input_b_out: torch.Tensor,
      input_b_down: torch.Tensor,
      input_b_up: torch.Tensor,
      input_s: torch.Tensor,
      input_interior: torch.Tensor,
      in_velocity: torch.Tensor,
      mu: float, rho: float
  ) -> torch.Tensor:

    input_f = utils.stack_xy_tensors(x_f, y_f)

    output_f = self(input_f)
    u = output_f[:, 0]
    v = output_f[:, 1]
    p = output_f[:, 2]

    u_x = utils.grad(u, x_f)
    u_y = utils.grad(u, y_f)
    u_xx = utils.grad(u_x, x_f, False)
    u_yy = utils.grad(u_y, y_f, False)

    v_x = utils.grad(v, x_f)
    v_y = utils.grad(v, y_f)
    v_xx = utils.grad(v_x, x_f, False)
    v_yy = utils.grad(v_y, y_f, False)

    p_x = utils.grad(p, x_f)
    p_xx = utils.grad(p_x, x_f, False)
    p_y = utils.grad(p, y_f)
    p_yy = utils.grad(p_y, y_f, False)

    b = self.__compute_b(u=u, v=v, 
                         u_x=u_x, u_y=u_y, 
                         v_x=v_x, v_y=v_y, 
                         rho=rho)

    # PDE loss
    ## Navier-Stokes equations
    ### X-momentum equation
    f1 = u*u_x + v*u_y + (1/rho) * p_x - mu * (u_xx + u_yy)
    ### Y-momentum equation
    f2 = u*v_x + v*v_y + (1/rho) * p_y - mu * (v_xx + v_yy)
    ### Continuity equation
    f3 = u_x + v_y
    # ### Poisson equation
    f4 = p_xx + p_yy - b

    pde_ns_loss = torch.mean(torch.square(f1)) + \
                  torch.mean(torch.square(f2)) + \
                  torch.mean(torch.square(f3))

    pde_ps_loss = torch.mean(torch.square(f4))

    # Boundary conditions loss
    ## Inlet: u = in_velocity, v = 0 & p = 1 for x = x_min
    output_b_in = self(input_b_in)
    u_b_in_pred = output_b_in[:, 0]
    v_b_in_pred = output_b_in[:, 1]
    u_b_in_true = torch.full_like(u_b_in_pred, in_velocity)
    bc_in_loss_u = torch.mean(torch.square(u_b_in_pred - u_b_in_true))
    bc_in_loss_v = torch.mean(torch.square(v_b_in_pred))
    bc_in_loss = bc_in_loss_u + bc_in_loss_v

    ## Outlet: p = 1 for x = x_max
    output_b_out = self(input_b_out)
    p_b_out_pred = output_b_out[:, 2]
    p_b_out_true = torch.ones_like(p_b_out_pred)
    bc_out_loss_p = torch.mean(torch.square(p_b_out_pred - p_b_out_true))
    bc_out_loss = bc_out_loss_p

    ## Down (Slip): v = 0 for y = y_min
    output_b_down = self(input_b_down)
    v_b_down_pred = output_b_down[:, 1]
    bc_down_loss_v = torch.mean(torch.square(v_b_down_pred))
    bc_down_loss = bc_down_loss_v

    ## Up (Slip): v = 0 for y = y_max
    output_b_up = self(input_b_up)
    v_b_up_pred = output_b_up[:, 1]
    bc_up_loss_v = torch.mean(torch.square(v_b_up_pred))
    bc_up_loss = bc_up_loss_v

    # Object surface boundary conditions loss
    output_s = self(input_s)
    u_s_pred = output_s[:, 0]
    v_s_pred = output_s[:, 1]

    ## no-slip condition in surface
    surface_loss_u = torch.mean(torch.square(u_s_pred))
    surface_loss_v = torch.mean(torch.square(v_s_pred))
    surface_loss = surface_loss_u + surface_loss_v

    # Interior points loss
    output_interior = self(input_interior)
    u_interior_pred = output_interior[:, 0]
    v_interior_pred = output_interior[:, 1]

    ## no-slip condition inside the airfoil
    interior_loss = torch.mean(torch.square(u_interior_pred)) + \
                    torch.mean(torch.square(v_interior_pred))

    # total loss
    total_loss =  self.lambda_pde_ns    * pde_ns_loss + \
                  self.lambda_pde_ps    * pde_ps_loss + \
                  self.lambda_bc_in     * bc_in_loss + \
                  self.lambda_bc_out    * bc_out_loss + \
                  self.lambda_bc_down   * bc_down_loss + \
                  self.lambda_bc_up     * bc_up_loss + \
                  self.lambda_surface   * surface_loss + \
                  self.lambda_interior  * interior_loss

    return total_loss, pde_ns_loss, pde_ns_loss, \
           bc_in_loss, bc_out_loss, \
           bc_down_loss, bc_up_loss, \
           surface_loss, interior_loss

  def closure(
      self, 
      optimizer: torch.optim.Optimizer, 
      Nf: int, Nb: int, Ns: int, Nin: int,
      x_min: float, y_min: float, 
      x_max: float, y_max: float, 
      in_velocity: int, 
      mu: float, rho: float, 
      device: torch.device
  ) -> torch.Tensor:

    optimizer.zero_grad()

    training_input = self.__generate_inputs(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, Nf=Nf, Nb=Nb, Ns=Ns, Nin=Nin, device=device)

    total_loss, pde_ns_loss, pde_ps_loss, bc_in_loss, bc_out_loss, bc_down_loss, bc_up_loss, surface_loss, interior_loss = self.loss(
                    *training_input,
                    in_velocity,
                    mu, rho)

    self.current_total_loss = total_loss.item()
    self.current_pde_ns_loss = pde_ns_loss.item()
    self.current_pde_ps_loss = pde_ps_loss.item()
    self.current_bc_in_loss = bc_in_loss.item()
    self.current_bc_out_loss = bc_out_loss.item()
    self.current_bc_down_loss = bc_down_loss.item()
    self.current_bc_up_loss = bc_up_loss.item()
    self.current_surface_loss = surface_loss.item()
    self.current_interior_loss = interior_loss.item()

    total_loss.backward()

    return total_loss


  def train_pinn(
        self, 
        epochs: int, 
        optimizer: torch.optim.Optimizer, 
        Nf: int, Nb: int, Ns: int, Nin: int,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
        in_velocity: int,
        mu: float, rho: float,
        device: torch.device,
        checkpoint_epochs: int,
        model_dir: str):

    print("=======================================================")
    print(self)
    print(f"Model name: {self.model_name}")
    print(f"Number of epochs: {epochs}")
    print(f"Number of collocation points Nf: {Nf}")
    print(f"Number of boundary condition points Nb: {Nb}")
    print(f"Number of object surface points Ns: {Ns}")
    print(f"Number of interior object points Ns: {Nin}")
    print(f"X min: {x_min}, Y min: {y_min}")
    print(f"X max: {x_max}, Y max: {y_max}")
    print(f"mu: {mu}, rho: {rho}")
    print(f"Inflow velocity: {in_velocity}")
    print(f"Device: {device}")
    print(f"Checkpoint epochs: {checkpoint_epochs}")
    print(f"Model directory: {model_dir}")
    print("=======================================================")
    print("=> converting Nf, Nb, Ns to nearest power of 2...") # Quasi - Monte Carlo sampling requirement

    Nf = utils.nearest_power_of_2(Nf)
    Nb = utils.nearest_power_of_2(Nb)
    Ns = utils.nearest_power_of_2(Ns)
    Nin = utils.nearest_power_of_2(Nin)

    print(f"Nf: {Nf}, Nb: {Nb}, Ns: {Ns}, Nin: {Nin}")
    print("=======================================================")
    print("=> starting training...")
    print("=======================================================")


    training_clock = utils.Clock()
    training_clock.start()

    relobralo = utils.ReLoBRaLo()

    try:
      while self.epoch < epochs:

        epoch_clock = utils.Clock()
        epoch_clock.start()

        self.epoch += 1

        optimizer.step(lambda: 
                      self.closure(
                        optimizer=optimizer, 
                        Nf=Nf, Nb=Nb, Ns=Ns, Nin=Nin,
                        x_min=x_min, y_min=y_min,
                        x_max=x_max, y_max=y_max,
                        in_velocity=in_velocity,
                        mu=mu, rho=rho,
                        device=device))

        self.__log_metrics(self.current_total_loss, 
                           self.current_pde_ns_loss, self.current_pde_ps_loss, 
                           self.current_bc_in_loss, self.current_bc_out_loss, 
                           self.current_bc_down_loss, self.current_bc_up_loss, 
                           self.current_surface_loss, self.current_interior_loss)

        self.print_current_metrics() 

        self.__update_lambdas(relobralo)

        self.__log_lambdas(self.lambda_pde_ns, self.lambda_pde_ps,
                          self.lambda_bc_in, self.lambda_bc_out, 
                          self.lambda_bc_down, self.lambda_bc_up, 
                          self.lambda_surface, self.lambda_interior)

        epoch_clock.stop()
        print(f"\t{epoch_clock}")

        if np.isnan(self.current_total_loss):
          print("=> NaN loss...")
          self, optimizer = self.load_last_checkpoint(optimizer, model_dir)
          continue

        if self.epoch % checkpoint_epochs == 0:
          checkpoint_path = os.path.join(model_dir, self.model_name, str(self.epoch) + ".pt")

          if not os.path.exists(os.path.dirname(checkpoint_path)):
            print(f"=> creating checkpoint directory at {os.path.dirname(checkpoint_path)}")
            os.makedirs(os.path.dirname(checkpoint_path))

          self.__save_checkpoint(optimizer, checkpoint_path)

      training_clock.stop()

      print("\n=======================================================")
      print(f"=> training completed.")
      print(f"{training_clock}")
      print("=======================================================")

    except KeyboardInterrupt:
      training_clock.stop()
      self.epoch -= 1

      print("\n=======================================================")
      print(f"=> training stopped by user at epoch {self.epoch}.")
      print(f"{training_clock}")
      print("=======================================================")


  def __compute_b(
        self, 
        u: torch.Tensor, v: torch.Tensor, 
        u_x: torch.Tensor, u_y: torch.Tensor, 
        v_x: torch.Tensor, v_y: torch.Tensor, 
        rho: torch.Tensor) -> torch.Tensor:

    convective_acc = u * u_x + v * u_y + u * v_x + v * v_y

    return rho * convective_acc


  def __generate_inputs(
        self, 
        x_min: float , x_max: float, y_min: float, y_max: float,  
        Nf: int, Nb: int, Ns: int, Nin: int,
        device: torch.device) -> tuple:

    # collocation points
    samples_f = utils.qmc_sample_points_in_domain_2d(_x_min=x_min, _x_max=x_max, 
                                                     _y_min=y_min, _y_max=y_max, 
                                                     num_samples=Nf/2)

    samples_f_near_the_airfoil = self.airfoil.sample_points_in_domain_around(Nf, 0.2*self.airfoil.chord)

    samples_f = np.concatenate((samples_f, samples_f_near_the_airfoil), axis=1)

    # filter out the points that are inside the airfoil
    _, exterior_samples = self.airfoil.classify_points(np.array(samples_f).T)

    x_f = utils.tensor_from_array(exterior_samples[:, 0], device=device, requires_grad=True)
    y_f = utils.tensor_from_array(exterior_samples[:, 1], device=device, requires_grad=True)

    # boundary condition points
    ## inflow, x=x_min
    Nb_in = utils.nearest_power_of_2(int(Nb/6))
    samples_b_in = utils.qmc_sample_points_in_domain_1d(_x_min=y_min, _x_max=y_max,
                                                        num_samples=Nb_in)

    x_b_in = utils.tensor_from_array(utils.full(Nb_in, x_min), device=device, requires_grad=False)
    y_b_in = utils.tensor_from_array(samples_b_in, device=device, requires_grad=False)
    xy_b_in = utils.stack_xy_tensors(x_b_in, y_b_in)

    ## outflow, x=x_max
    Nb_out = utils.nearest_power_of_2(int(Nb/6))
    samples_b_out = utils.qmc_sample_points_in_domain_1d(_x_min=y_min, _x_max=y_max,
                                                        num_samples=Nb_out)

    x_b_out = utils.tensor_from_array(utils.full(Nb_in, x_max), device=device, requires_grad=False)
    y_b_out = utils.tensor_from_array(samples_b_out, device=device, requires_grad=False)
    xy_b_out = utils.stack_xy_tensors(x_b_out, y_b_out)

    ## down, y=y_min
    Nb_down = utils.nearest_power_of_2(int(Nb/6))
    samples_b_down = utils.qmc_sample_points_in_domain_1d(_x_min=x_min, _x_max=x_max,
                                                          num_samples=Nb_down)

    x_b_down = utils.tensor_from_array(samples_b_down, device=device, requires_grad=False)
    y_b_down = utils.tensor_from_array(utils.full(Nb_down, y_min), device=device, requires_grad=False)
    xy_b_down = utils.stack_xy_tensors(x_b_down, y_b_down)

    ## up, y=y_max
    Nb_up = utils.nearest_power_of_2(int(Nb/6))
    samples_b_up = utils.qmc_sample_points_in_domain_1d(_x_min=x_min, _x_max=x_max,
                                                        num_samples=Nb_up)

    x_b_up = utils.tensor_from_array(samples_b_up, device=device, requires_grad=False)
    y_b_up = utils.tensor_from_array(utils.full(Nb_up, y_max), device=device, requires_grad=False)
    xy_b_up = utils.stack_xy_tensors(x_b_up, y_b_up)

    # points on the surface of the airfoil
    surface_points = self.airfoil.sample_surface_points(Ns)
    x_s = utils.tensor_from_array(surface_points[0], device=device, requires_grad=False)
    y_s = utils.tensor_from_array(surface_points[1], device=device, requires_grad=False)

    xy_s = utils.stack_xy_tensors(x_s, y_s)

    # points inside the airfoil
    interior_points = self.airfoil.generate_interior_points(Nin)
    x_in = utils.tensor_from_array(interior_points[0], device=device, requires_grad=False)
    y_in = utils.tensor_from_array(interior_points[1], device=device, requires_grad=False)

    xy_in = utils.stack_xy_tensors(x_in, y_in)

    return (x_f, y_f, xy_b_in, xy_b_out, xy_b_down, xy_b_up, xy_s, xy_in)


  def __update_lambdas(self, relobralo: utils.ReLoBRaLo):
    """ Update loss lambdas """
    losses = [self.logs["pde_ns_loss"], self.logs["pde_ps_loss"], 
              self.logs["bc_in_loss"], self.logs["bc_out_loss"], 
              self.logs["bc_down_loss"], self.logs["bc_up_loss"], 
              self.logs["surface_loss"], self.logs["interior_loss"]]

    lambdas = relobralo.compute_next_lambdas(L=losses)

    self.lambda_pde_ns    = lambdas[0]
    self.lambda_pde_ps    = lambdas[1]
    self.lambda_bc_in     = lambdas[2]
    self.lambda_bc_out    = lambdas[3]
    self.lambda_bc_down   = lambdas[4]
    self.lambda_bc_up     = lambdas[5]
    self.lambda_surface   = lambdas[6]
    self.lambda_interior  = lambdas[7]


  def __log_metrics(self, total_loss: float, 
                    pde_ns_loss: float, pde_ps_loss: float,
                    bc_in_loss: float, bc_out_loss: float, 
                    bc_down_loss: float, bc_up_loss: float, 
                    surface_loss: float, interior_loss: float):
    """ Log training metrics """
    self.logs['total_loss'].append(total_loss)
    self.logs['pde_ns_loss'].append(pde_ns_loss)
    self.logs['pde_ps_loss'].append(pde_ps_loss)
    self.logs['bc_in_loss'].append(bc_in_loss)
    self.logs['bc_out_loss'].append(bc_out_loss)
    self.logs['bc_down_loss'].append(bc_down_loss)
    self.logs['bc_up_loss'].append(bc_up_loss)
    self.logs['surface_loss'].append(surface_loss)
    self.logs['interior_loss'].append(interior_loss)


  def __log_lambdas(self, lambda_pde_ns: float, lambda_pde_ps: float,
                    lambda_bc_in: float, lambda_bc_out: float, 
                    lambda_bc_down: float, lambda_bc_up: float, 
                    lambda_surface: float, lambda_interior: float):
      """ Log loss lambdas """
      self.lambdas['pde_ns'].append(lambda_pde_ns)
      self.lambdas['pde_ps'].append(lambda_pde_ps)
      self.lambdas['bc_in'].append(lambda_bc_in)
      self.lambdas['bc_out'].append(lambda_bc_out)
      self.lambdas['bc_down'].append(lambda_bc_down)
      self.lambdas['bc_up'].append(lambda_bc_up)
      self.lambdas['surface'].append(lambda_surface)
      self.lambdas['interior'].append(lambda_interior)


  def __get_logs(self):
    """ Retrieve the logged metrics """
    return self.logs


  def print_current_metrics(self):
      """ Print the most recent set of metrics """
      if self.logs['total_loss']:
          print(f"\nEpoch: {self.epoch}\n"
                f"\tTotal Loss: {self.logs['total_loss'][-1]:.4f}\n"
                f"\tPDE Loss - Navier Stokes: {self.logs['pde_ns_loss'][-1]:.4f}, "
                f"PDE Loss - Poisson: {self.logs['pde_ps_loss'][-1]:.4f}, "
                f"BC Inlet Loss: {self.logs['bc_in_loss'][- 1]:.4f}, "
                f"BC Outlet Loss: {self.logs['bc_out_loss'][- 1]:.4f}, "
                f"BC Down Loss: {self.logs['bc_down_loss'][- 1]:.4f}, "
                f"BC Up Loss: {self.logs['bc_up_loss'][- 1]:.4f}, "
                f"Surface Loss: {self.logs['surface_loss'][-1]:.4f},"
                f"Interior Loss: {self.logs['interior_loss'][-1]:.4f}")
      else:
          print("No metrics to display.")


  def print_all_metrics(self):
      """ Print all metrics """

      for _epoch in range(1, self.epoch + 1): 
      
        if self.logs['total_loss']:
            print(f"\nEpoch: {_epoch}\n"
                  f"\tTotal Loss: {self.logs['total_loss'][_epoch - 1]:.4f}\n"
                  f"\tPDE Loss - Navier Stokes: {self.logs['pde_ns_loss'][_epoch - 1]:.4f}, "
                  f"PDE Loss - Poisson: {self.logs['pde_ps_loss'][_epoch - 1]:.4f}, "
                  f"BC Inlet Loss: {self.logs['bc_in_loss'][_epoch - 1]:.4f}, "
                  f"BC Outlet Loss: {self.logs['bc_out_loss'][_epoch - 1]:.4f}, "
                  f"BC Down Loss: {self.logs['bc_down_loss'][_epoch - 1]:.4f}, "
                  f"BC Up Loss: {self.logs['bc_up_loss'][_epoch - 1]:.4f}, "
                  f"Surface Loss: {self.logs['surface_loss'][_epoch - 1]:.4f},"
                  f"Interior Loss: {self.logs['interior_loss'][_epoch - 1]:.4f}")
        else:
            print("No metrics to display.")


  def __save_checkpoint(self, optimizer: torch.optim.Optimizer, file_path: str):

    print("=> saving checkpoint '{}'".format(file_path))
    state = {'name': self.model_name, 'input_dim': self.input_dim, 'output_dim': self.output_dim, 'hidden_units': self.hidden_units, 'epoch': self.epoch, 'state_dict': self.state_dict(),
              'optimizer': optimizer.state_dict(), "logs": self.logs, "lambdas": self.lambdas}
    torch.save(state, file_path)


  def __load_checkpoint(self, optimizer: torch.optim.Optimizer, file_path: str, mode='training') -> 'AirfoilPINN, torch.optim.Optimizer':

      if os.path.isfile(file_path):
          print("=> loading checkpoint '{}'".format(file_path))
          checkpoint = torch.load(file_path)

          if self.input_dim != checkpoint['input_dim'] or self.output_dim != checkpoint['output_dim'] or self.hidden_units != checkpoint['hidden_units']:
            print("=> model architecture mismatch, exiting...")
            exit

          self.load_state_dict(checkpoint['state_dict'])
          self.model_name = checkpoint['name']
          self.epoch = checkpoint['epoch']
          self.logs = checkpoint['logs']
          self.lambdas = checkpoint['lambdas']
          
          if mode == 'training':
            optimizer.load_state_dict(checkpoint['optimizer'])

          print("=> loaded checkpoint '{}' (epoch {})"
                    .format(file_path, checkpoint['epoch']))
      else:
          print("=> no checkpoint found at '{}'".format(file_path))

      return self, optimizer


  def load_last_checkpoint(self, optimizer: torch.optim.Optimizer, checkpoint_dir: str) -> 'AirfoilPINN, torch.optim.Optimizer':

    # Get a list of all files and directories in the specified directory
    all_items = os.listdir(os.path.join(checkpoint_dir, self.model_name))

    # Filter out directories, keep only files
    checkpoint_files = [item for item in all_items if os.path.isfile(os.path.join(checkpoint_dir, self.model_name, item))]

    # Sort the files based on the numeric part of the file name
    checkpoint_files.sort(key=lambda x: int(x.split('.')[0]), reverse=True)

    # Now, 'files' contains all files in the directory, sorted alphabetically
    checkpoint_file_name = checkpoint_files[0]

    checkpoint_file_path = os.path.join(checkpoint_dir, self.model_name, checkpoint_file_name)

    self, optimizer = self.__load_checkpoint(optimizer, checkpoint_file_path)

    return self, optimizer


  def load_checkpoint_num(self, optimizer: torch.optim.Optimizer, checkpoint_dir, model_name: str, checkpoint_num: int, mode='training') -> 'AirfoilPINN, torch.optim.Optimizer':

    file_path = os.path.join(checkpoint_dir, model_name, str(checkpoint_num) + ".pt")
    return self.__load_checkpoint(optimizer, file_path, mode)
  
  @staticmethod
  def load_from_checkpoint_for_testing(checkpoint_dir: str, model_name: str, checkpoint_num: int) -> 'AirfoilPINN':

    file_path = os.path.join(checkpoint_dir, model_name, str(checkpoint_num) + ".pt")

    if os.path.isfile(file_path):
      print("=> loading checkpoint '{}'".format(file_path))
      checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

      _input_dim = checkpoint['input_dim']
      _output_dim = checkpoint['output_dim']
      _hidden_units = checkpoint['hidden_units']
 
      _pinn = AirfoilPINN(hidden_units=_hidden_units, model_name=model_name, airfoil=None)

      _pinn.epoch = checkpoint['epoch']
      _pinn.logs = checkpoint['logs']
      _pinn.lambdas = checkpoint['lambdas']

      _pinn.load_state_dict(checkpoint['state_dict'])

      print("=> loaded checkpoint '{}' (epoch {})"
                .format(file_path, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(file_path))
      return None

    _pinn.eval()

    return _pinn

  @staticmethod
  def load_from_checkpoint_for_training(checkpoint_dir: str, model_name: str, checkpoint_num: int, device: torch.device,lr=1) -> ('AirfoilPINN', torch.optim.Optimizer):

    file_path = os.path.join(checkpoint_dir, model_name, str(checkpoint_num) + ".pt")

    if os.path.isfile(file_path):
      print("=> loading checkpoint '{}'".format(file_path))
      checkpoint = torch.load(file_path, map_location=torch.device(device))

      _hidden_units = checkpoint['hidden_units']
 
      _pinn = AirfoilPINN(hidden_units=_hidden_units, model_name=model_name)

      _pinn.epoch = checkpoint['epoch']
      _pinn.logs = checkpoint['logs']
      _pinn.lambdas = checkpoint['lambdas']

      _pinn.load_state_dict(checkpoint['state_dict'])

      _optimizer = torch.optim.LBFGS(_pinn.parameters(), lr=lr, line_search_fn="strong_wolfe")

      _optimizer.load_state_dict(checkpoint['optimizer'])

      print("=> loaded checkpoint '{}' (epoch {})"
                .format(file_path, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(file_path))
      return None

    _pinn.eval()

    return _pinn, _optimizer


  def plot_learning_curves(self, output_dir: str, save: bool = False):
    fig, axs = plt.subplots(4, 2, figsize=(20, 15))

    linewidth = 0.5

    axs[0, 0].plot(self.logs['pde_ns_loss'], linewidth=linewidth)
    axs[0, 0].set_title('PDE loss - Navier Stokes')

    axs[0, 1].plot(self.logs['pde_ps_loss'], linewidth=linewidth)
    axs[0, 1].set_title('PDE loss - Poisson')

    axs[1, 0].plot(self.logs['bc_in_loss'], linewidth=linewidth)
    axs[1, 0].set_title('BC loss - Inlet')

    axs[1, 1].plot(self.logs['bc_out_loss'], linewidth=linewidth)
    axs[1, 1].set_title('BC loss - Outlet')

    axs[2, 0].plot(self.logs['bc_down_loss'], linewidth=linewidth)
    axs[2, 0].set_title('BC loss - Down')

    axs[2, 1].plot(self.logs['bc_up_loss'], linewidth=linewidth)
    axs[2, 1].set_title('BC loss - Up')

    axs[3, 0].plot(self.logs['surface_loss'], linewidth=linewidth)
    axs[3, 0].set_title('Surface loss')

    axs[3, 1].plot(self.logs['interior_loss'], linewidth=linewidth)
    axs[3, 1].set_title('Interior loss')

    _output_dir = os.path.join(output_dir, self.model_name)

    # if save and os.path.exists(os.path.dirname(_output_dir)):
      # print(f"=> saving learning curves plot at {os.path.dirname(_output_dir)}")
      # plt.savefig(os.path.join(_output_dir, "fig/learning_curves.png"))

  def plot_lambdas(self):
    fig, axs = plt.subplots(4, 2, figsize=(20, 15))

    linewidth = 0.5

    axs[0, 0].plot(self.lambdas['pde_ns'], linewidth=linewidth)
    axs[0, 0].set_title('lambda PDE - Navier Stokes')

    axs[0, 0].plot(self.lambdas['pde_pa'], linewidth=linewidth)
    axs[0, 0].set_title('lambda PDE - Poisson')

    axs[1, 0].plot(self.lambdas['bc_in'], linewidth=linewidth)
    axs[1, 0].set_title('lambda BC - Inlet')

    axs[1, 1].plot(self.lambdas['bc_out'], linewidth=linewidth)
    axs[1, 1].set_title('lambda BC - Outlet')

    axs[2, 0].plot(self.lambdas['bc_down'], linewidth=linewidth)
    axs[2, 0].set_title('lambda BC - Down')

    axs[2, 1].plot(self.lambdas['bc_up'], linewidth=linewidth)
    axs[2, 1].set_title('lambda BC - Up')

    axs[3, 0].plot(self.lambdas['surface'], linewidth=linewidth)
    axs[3, 0].set_title('lambda Surface')

    axs[3, 1].plot(self.lambdas['interior'], linewidth=linewidth)
    axs[3, 1].set_title('lambda Interior')

    # for i in range(1, 7):
    #   axs[2, 2].plot(self.lambdas[list(self.lambdas.keys())[i]], linewidth=linewidth)
    # axs[2, 2].set_title('All lambdas')
