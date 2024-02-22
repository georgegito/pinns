import torch
import torch.nn as nn
import numpy as np
import utils
import os
import pandas as pd
from matplotlib import pyplot as plt
from naca4digit_airfoil import Naca4DigitAirfoil

class AirfoilPINN(nn.Module):

  def __init__(self, hidden_units: int, activation_function: str , airfoil: Naca4DigitAirfoil, domain: utils.Domain2D, u_in: float, p_out: float, model_name: str = None):

    super(AirfoilPINN, self).__init__()

    self.airfoil = airfoil

    self.layers = nn.ModuleList()

    self.input_dim = 2
    self.output_dim = 3

    self.activation_function = activation_function

    self.domain = domain
    self.u_in = u_in
    self.p_out = p_out

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
                 "pde_ns_loss": [], #"pde_ps_loss": [],
                 "bc_in_loss": [], "bc_out_loss": [],
                 "bc_down_loss": [], "bc_up_loss": [], 
                 "surface_loss": [], "interior_loss": [],
                 "data_loss": []}

    self.lambdas = {"pde_ns": [], #"pde_ps": [], 
                   "bc_in": [], "bc_out": [],
                   "bc_down": [], "bc_up": [], 
                   "surface": [], "interior": [],
                   "data": []}

    self.curent_total_loss = -1
    self.current_pde_nv_loss = -1
    # self.current_pde_ps_loss = -1
    self.current_bc_in_loss = -1
    self.current_bc_out_loss = -1
    self.current_bc_down_loss = -1
    self.current_bc_up_loss = -1
    self.current_surface_loss = -1
    self.current_interior_loss = -1
    self.current_data_loss = -1

    self.epoch = 0
    self.hidden_units = hidden_units

    if model_name is None:
      self.model_name = utils.NameGenerator().generate_name()
    else:
      self.model_name = model_name

    self.lambda_pde_ns = .1
    # self.lambda_pde_ps = .1
    self.lambda_bc_in = .1
    self.lambda_bc_out = .1
    self.lambda_bc_down = .1
    self.lambda_bc_up = .1
    self.lambda_surface = .1
    self.lambda_interior = .1
    self.lambda_data = .1


  def forward(self, x: torch.Tensor) -> torch.Tensor:

    # input layer
    x = self.activation_function(self.layers[0](x))
    # x = nn.BatchNorm1d(x.size(1))(x)

    # hidden layers
    for layer in self.layers[1:-1]:
      x = self.activation_function(layer(x))
      # x = nn.BatchNorm1d(x.size(1))(x)

    # output layer
    x = self.activation_function(self.layers[-1](x))

    return x


  def mse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(y_true - y_pred))


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
      training_data_xy: torch.Tensor,
      training_data_p: torch.Tensor,
      mu: float, rho: float
  ) -> torch.Tensor:

    input_f = utils.stack_xy_tensors(x_f, y_f)

    inputs = [input_f, input_b_in, input_b_out, input_b_down, input_b_up, input_s, input_interior, training_data_xy]
    concatenated_inputs = torch.cat(inputs, dim=0)
    concatenated_outputs = self(concatenated_inputs)
    split_sizes = [input.size(0) for input in inputs]
    outputs = torch.split(concatenated_outputs, split_sizes, dim=0)
    output_f, output_b_in, output_b_out, output_b_down, output_b_up, output_s, output_interior, output_data = outputs

    # output_f = self(input_f)
    # output_b_in = self(input_b_in)
    # output_b_out = self(input_b_out)
    # output_b_down = self(input_b_down)
    # output_b_up = self(input_b_up)
    # output_s = self(input_s)
    # output_interior = self(input_interior)

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
    # p_xx = utils.grad(p_x, x_f, False)
    p_y = utils.grad(p, y_f)
    # p_yy = utils.grad(p_y, y_f, False)

    # PDE loss
    ## Navier-Stokes equations
    ### X-momentum equation
    f1 = u*u_x + v*u_y + (1/rho) * p_x - mu * (u_xx + u_yy)
    ### Y-momentum equation
    f2 = u*v_x + v*v_y + (1/rho) * p_y - mu * (v_xx + v_yy)
    ### Continuity equation
    f3 = u_x + v_y
    ### Poisson equation
    # f4 = p_xx + p_yy - rho * (u * u_x + v * u_y + u * v_x + v * v_y)


    pde_ns_loss = self.mse(0, f1) + self.mse(0, f2) + self.mse(0, f3)

    # pde_ps_loss = torch.mean(torch.square(f4))

    # Boundary conditions loss
    ## Inlet: u = in_velocity, v = 0 & p = 1 for x = x_min
    u_b_in_pred = output_b_in[:, 0]
    v_b_in_pred = output_b_in[:, 1]
    u_b_in_true = torch.full_like(u_b_in_pred, self.u_in)
    bc_in_loss_u = self.mse(u_b_in_true, u_b_in_pred)
    bc_in_loss_v = self.mse(0, v_b_in_pred)

    bc_in_loss = bc_in_loss_u + bc_in_loss_v

    ## Outlet: p = out_pressure for x = x_max
    p_b_out_pred = output_b_out[:, 2]
    p_b_out_true = torch.full_like(p_b_out_pred, self.p_out)
    bc_out_loss_p = self.mse(p_b_out_true, p_b_out_pred)
    bc_out_loss = bc_out_loss_p

    ## Down (Slip): v = 0 for y = y_min
    v_b_down_pred = output_b_down[:, 1]
    bc_down_loss_v = self.mse(0, v_b_down_pred)
    bc_down_loss = bc_down_loss_v

    ## Up (Slip): v = 0 for y = y_max
    v_b_up_pred = output_b_up[:, 1]
    bc_up_loss_v = self.mse(0, v_b_up_pred)
    bc_up_loss = bc_up_loss_v

    # Object surface boundary conditions loss
    u_s_pred = output_s[:, 0]
    v_s_pred = output_s[:, 1]

    ## no-slip condition in surface
    surface_loss_u = self.mse(0, u_s_pred)
    surface_loss_v = self.mse(0, v_s_pred)

    surface_loss = surface_loss_u + surface_loss_v

    # Interior points loss
    u_interior_pred = output_interior[:, 0]
    v_interior_pred = output_interior[:, 1]

    ## no-slip condition inside the airfoil
    interior_loss = self.mse(0, u_interior_pred) + self.mse(0, v_interior_pred)

    ## data loss
    p_data_pred = output_data[:, 2]
    data_loss = self.mse(p_data_pred, training_data_p)

    # total loss
    total_loss =  self.lambda_pde_ns    * pde_ns_loss + \
                  self.lambda_bc_in     * bc_in_loss + \
                  self.lambda_bc_out    * bc_out_loss + \
                  self.lambda_bc_down   * bc_down_loss + \
                  self.lambda_bc_up     * bc_up_loss + \
                  self.lambda_surface   * surface_loss + \
                  self.lambda_interior  * interior_loss + \
                  self.lambda_data      * data_loss

    return total_loss, pde_ns_loss, \
           bc_in_loss, bc_out_loss, \
           bc_down_loss, bc_up_loss, \
           surface_loss, interior_loss, data_loss

  def closure(
      self, 
      optimizer: torch.optim.Optimizer, 
      Nf1: int, Nf2: int, Nf3: int, Nf4: int, Nb: int, Ns: int, Nin: int,
      training_data_xy: torch.Tensor, training_data_p: torch.Tensor,
      domain1: utils.Domain2D, domain2: utils.Domain2D, domain3: utils.Domain2D, domain4: utils.Domain2D,
      mu: float, rho: float, 
      device: torch.device
  ) -> torch.Tensor:

    optimizer.zero_grad()

    training_input = self.__generate_inputs(domain1=domain1, domain2=domain2, domain3=domain3, domain4=domain4,
                                            Nf1=Nf1, Nf2=Nf2, Nf3=Nf3, Nf4=Nf4,
                                            Nb=Nb, Ns=Ns, Nin=Nin, device=device)

    total_loss, pde_ns_loss, bc_in_loss, bc_out_loss, bc_down_loss, bc_up_loss, surface_loss, interior_loss, data_loss = self.loss(
                    *training_input, 
                    training_data_xy=training_data_xy, 
                    training_data_p=training_data_p,
                    mu=mu, rho=rho)

    self.current_total_loss = total_loss.item()
    self.current_pde_ns_loss = pde_ns_loss.item()
    # self.current_pde_ps_loss = pde_ps_loss.item()
    self.current_bc_in_loss = bc_in_loss.item()
    self.current_bc_out_loss = bc_out_loss.item()
    self.current_bc_down_loss = bc_down_loss.item()
    self.current_bc_up_loss = bc_up_loss.item()
    self.current_surface_loss = surface_loss.item()
    self.current_interior_loss = interior_loss.item()
    self.current_data_loss = data_loss.item()

    total_loss.backward()

    return total_loss


  def train_pinn(
        self, 
        epochs: int, 
        optimizer: torch.optim.Optimizer, 
        Nf1: int, Nf2: int, Nf3: int, Nf4:int, Nb: int, Ns: int, Nin: int,
        training_data: pd.DataFrame,
        domain1: utils.Domain2D, domain2: utils.Domain2D, domain3: utils.Domain2D, domain4: utils.Domain2D,
        mu: float, rho: float,
        device: torch.device,
        checkpoint_epochs: int,
        model_dir: str):

    print("=======================================================")
    print(self)
    print(f"Model name: {self.model_name}")
    print(f"Number of epochs: {epochs}")
    print(f"Number of collocation points Nf: {Nf1 + Nf2 + Nf3 + Nf4}")
    print(f"Number of boundary condition points Nb: {Nb}")
    print(f"Number of object surface points Ns: {Ns}")
    print(f"Number of interior object points Nin: {Nin}")
    print(f"X min: {domain1.x_min}, Y min: {domain1.y_min}")
    print(f"X max: {domain1.x_max}, Y max: {domain1.y_max}")
    print(f"mu: {mu}, rho: {rho}")
    print(f"Inflow velocity: {self.u_in}")
    print(f"Outflow pressure: {self.p_out}")
    print(f"Device: {device}")
    print(f"Checkpoint epochs: {checkpoint_epochs}")
    print(f"Model directory: {model_dir}")
    print("=======================================================")
    print("=> converting Nf1, Nf2, Nb, Ns to nearest power of 2...") # Quasi - Monte Carlo sampling requirement

    Nf1 = utils.nearest_power_of_2(Nf1)
    Nf2 = utils.nearest_power_of_2(Nf2)
    Nf3 = utils.nearest_power_of_2(Nf3)
    Nf4 = utils.nearest_power_of_2(Nf4)
    Nb = utils.nearest_power_of_2(Nb)
    Ns = utils.nearest_power_of_2(Ns)
    Nin = utils.nearest_power_of_2(Nin)

    print(f"Nf1: {Nf1}, Nf2: {Nf2}, Nf3: {Nf3}, Nf4: {Nf4}, Nb: {Nb}, Ns: {Ns}, Nin: {Nin}")
    print("=======================================================")
    print("=> starting training...")
    print("=======================================================")

    training_data_xy = utils.stack_xy_tensors(
      utils.tensor_from_array(training_data['x'], device=device, requires_grad=False), 
      utils.tensor_from_array(training_data['y'], device=device, requires_grad=False)) 
    training_data_p = utils.tensor_from_array(training_data['p'], device=device, requires_grad=False)

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
                        Nf1=Nf1, Nf2=Nf2, Nf3=Nf3, Nf4=Nf4, Nb=Nb, Ns=Ns, Nin=Nin,
                        training_data_xy=training_data_xy, training_data_p=training_data_p,
                        domain1=domain1, domain2=domain2, domain3=domain3, domain4=domain4,
                        mu=mu, rho=rho,
                        device=device))

        self.__log_metrics(self.current_total_loss, 
                           self.current_pde_ns_loss, #self.current_pde_ps_loss, 
                           self.current_bc_in_loss, self.current_bc_out_loss, 
                           self.current_bc_down_loss, self.current_bc_up_loss, 
                           self.current_surface_loss, self.current_interior_loss,
                           self.current_data_loss)

        self.print_current_metrics() 

        self.__update_lambdas(relobralo)

        self.__log_lambdas(self.lambda_pde_ns, #self.lambda_pde_ps,
                          self.lambda_bc_in, self.lambda_bc_out,
                          self.lambda_bc_down, self.lambda_bc_up, 
                          self.lambda_surface, self.lambda_interior,
                          self.lambda_data)

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


  def __generate_inputs(
        self, 
        domain1: utils.Domain2D, domain2: utils.Domain2D, domain3: utils.Domain2D, domain4: utils.Domain2D,
        Nf1: int, Nf2: int, Nf3: int, Nf4: int, Nb: int, Ns: int, Nin: int,
        device: torch.device) -> tuple:
    
    x_min, x_max, y_min, y_max = domain1.x_min, domain1.x_max, domain1.y_min, domain1.y_max

    # collocation points
    samples_f1 = utils.qmc_sample_points_in_domain_2d(domain=domain1, num_samples=Nf1)
    samples_f2 = utils.qmc_sample_points_in_domain_2d(domain=domain2, num_samples=Nf2)
    samples_f3 = utils.qmc_sample_points_in_domain_2d(domain=domain3, num_samples=Nf3)
    samples_f4 = utils.qmc_sample_points_in_domain_2d(domain=domain4, num_samples=Nf4)

    samples_f = np.concatenate((samples_f1, samples_f2, samples_f3, samples_f4), axis=1)

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
    losses = [self.logs["pde_ns_loss"], #self.logs["pde_ps_loss"], 
              self.logs["bc_in_loss"], self.logs["bc_out_loss"],
              self.logs["bc_down_loss"], self.logs["bc_up_loss"], 
              self.logs["surface_loss"], self.logs["interior_loss"],
              self.logs["data_loss"]]

    lambdas = relobralo.compute_next_lambdas(L=losses)

    self.lambda_pde_ns    = lambdas[0]
    # self.lambda_pde_ps    = lambdas[1]
    self.lambda_bc_in     = lambdas[1]
    self.lambda_bc_out    = lambdas[2]
    self.lambda_bc_down   = lambdas[3]
    self.lambda_bc_up     = lambdas[4]
    self.lambda_surface   = lambdas[5]
    self.lambda_interior  = lambdas[6]
    self.lambda_data      = lambdas[7]


  def __log_metrics(self, total_loss: float, 
                    pde_ns_loss: float, #pde_ps_loss: float,
                    bc_in_loss: float, bc_out_loss: float, 
                    bc_down_loss: float, bc_up_loss: float, 
                    surface_loss: float, interior_loss: float,
                    data_loss: float):
    """ Log training metrics """
    self.logs['total_loss'].append(total_loss)
    self.logs['pde_ns_loss'].append(pde_ns_loss)
    # self.logs['pde_ps_loss'].append(pde_ps_loss)
    self.logs['bc_in_loss'].append(bc_in_loss)
    self.logs['bc_out_loss'].append(bc_out_loss)
    self.logs['bc_down_loss'].append(bc_down_loss)
    self.logs['bc_up_loss'].append(bc_up_loss)
    self.logs['surface_loss'].append(surface_loss)
    self.logs['interior_loss'].append(interior_loss)
    self.logs['data_loss'].append(data_loss)


  def __log_lambdas(self, lambda_pde_ns: float, #lambda_pde_ps: float,
                    lambda_bc_in: float, lambda_bc_out: float,
                    lambda_bc_down: float, lambda_bc_up: float, 
                    lambda_surface: float, lambda_interior: float,
                    lambda_data: float):
      """ Log loss lambdas """
      self.lambdas['pde_ns'].append(lambda_pde_ns)
      #self.lambdas['pde_ps'].append(lambda_pde_ps)
      self.lambdas['bc_in'].append(lambda_bc_in)
      self.lambdas['bc_out'].append(lambda_bc_out)
      self.lambdas['bc_down'].append(lambda_bc_down)
      self.lambdas['bc_up'].append(lambda_bc_up)
      self.lambdas['surface'].append(lambda_surface)
      self.lambdas['interior'].append(lambda_interior)
      self.lambdas['data'].append(lambda_data)


  def print_current_metrics(self):
      """ Print the most recent set of metrics """
      if self.logs['total_loss']:
          print(f"\nEpoch: {self.epoch}\n"
                f"\tTotal Loss: {self.logs['total_loss'][-1]:.4f}\n"
                f"\tPDE Loss - Navier Stokes: {self.logs['pde_ns_loss'][-1]:.4f}, "
                # f"PDE Loss - Poisson: {self.logs['pde_ps_loss'][-1]:.4f}, "
                f"BC Inlet Loss: {self.logs['bc_in_loss'][- 1]:.4f}, "
                f"BC Outlet Loss: {self.logs['bc_out_loss'][- 1]:.4f}, "
                f"BC Down Loss: {self.logs['bc_down_loss'][- 1]:.4f}, "
                f"BC Up Loss: {self.logs['bc_up_loss'][- 1]:.4f}, "
                f"Surface Loss: {self.logs['surface_loss'][-1]:.4f}, "
                f"Interior Loss: {self.logs['interior_loss'][-1]:.4f}, "
                f"Data Loss: {self.logs['data_loss'][-1]:.4f}")
                
      else:
          print("No metrics to display.")


  def print_all_metrics(self):
      """ Print all metrics """

      for _epoch in range(1, self.epoch + 1): 
      
        if self.logs['total_loss']:
            print(f"\nEpoch: {_epoch}\n"
                  f"\tTotal Loss: {self.logs['total_loss'][_epoch - 1]:.4f}\n"
                  f"\tPDE Loss - Navier Stokes: {self.logs['pde_ns_loss'][_epoch - 1]:.4f}, "
                  # f"PDE Loss - Poisson: {self.logs['pde_ps_loss'][_epoch - 1]:.4f}, "
                  f"BC Inlet Loss: {self.logs['bc_in_loss'][_epoch - 1]:.4f}, "
                  f"BC Outlet Loss: {self.logs['bc_out_loss'][_epoch - 1]:.4f}, "
                  f"BC Down Loss: {self.logs['bc_down_loss'][_epoch - 1]:.4f}, "
                  f"BC Up Loss: {self.logs['bc_up_loss'][_epoch - 1]:.4f}, "
                  f"Surface Loss: {self.logs['surface_loss'][_epoch - 1]:.4f},"
                  f"Interior Loss: {self.logs['interior_loss'][_epoch - 1]:.4f},"
                  f"Data Loss: {self.logs['data_loss'][_epoch - 1]:.4f}")
        else:
            print("No metrics to display.")

  def __save_checkpoint(self, optimizer: torch.optim.Optimizer, file_path: str):

    print("=> saving checkpoint '{}'".format(file_path))
    state = {'name': self.model_name, 'input_dim': self.input_dim, 'output_dim': self.output_dim, 'hidden_units': self.hidden_units, 'activation_function': self.activation_function, 
             'u_in': self.u_in, 'p_out': self.p_out, 'x_min': self.domain.x_min, 'x_max': self.domain.x_max, 'y_min': self.domain.y_min, 'y_max': self.domain.y_max,
             'epoch': self.epoch, 'state_dict': self.state_dict(),
              'optimizer': optimizer.state_dict(), "logs": self.logs, "lambdas": self.lambdas}
    torch.save(state, file_path)


  def __load_checkpoint(self, optimizer: torch.optim.Optimizer, file_path: str, mode='training') -> 'AirfoilPINN, torch.optim.Optimizer':

      if os.path.isfile(file_path):
          print("=> loading checkpoint '{}'".format(file_path))
          checkpoint = torch.load(file_path)

          if self.input_dim != checkpoint['input_dim'] or self.output_dim != checkpoint['output_dim'] or self.hidden_units != checkpoint['hidden_units'] or self.activation_function != checkpoint['activation_function']:
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
      _activation_function = checkpoint['activation_function']
      _u_in = checkpoint['u_in']
      _p_out = checkpoint['p_out']
      _x_min = checkpoint['x_min']
      _x_max = checkpoint['x_max']
      _y_min = checkpoint['y_min']
      _y_max = checkpoint['y_max']
 
      _pinn = AirfoilPINN(hidden_units=_hidden_units, 
                          activation_function=_activation_function, 
                          model_name=model_name, 
                          domain=utils.Domain2D(x_min=_x_min, x_max=_x_max, y_min=_y_min, y_max=_y_max), 
                          u_in=_u_in, p_out=_p_out,
                          airfoil=None)

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
  def load_from_checkpoint_for_training(checkpoint_dir: str, model_name: str, checkpoint_num: int, device: torch.device, activation_function, airfoil, domain, u_in, p_out, lr=1) -> ('AirfoilPINN', torch.optim.Optimizer):

    file_path = os.path.join(checkpoint_dir, model_name, str(checkpoint_num) + ".pt")

    if os.path.isfile(file_path):
      print("=> loading checkpoint '{}'".format(file_path))
      checkpoint = torch.load(file_path, map_location=torch.device(device))

      _hidden_units = checkpoint['hidden_units']
 
      _pinn = AirfoilPINN(hidden_units=_hidden_units, model_name=model_name, activation_function=activation_function, airfoil=airfoil, domain=domain, u_in=u_in, p_out=p_out)

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

    # _pinn.eval()

    return _pinn, _optimizer


  def plot_learning_curves(self, output_dir: str, save: bool = False, start_index=0):
    fig, axs = plt.subplots(4, 2, figsize=(20, 15))

    linewidth = 0.5

    axs[0, 0].plot(self.logs['pde_ns_loss'][start_index:], linewidth=linewidth)
    axs[0, 0].set_title('PDE loss - Navier Stokes')

    axs[0, 1].plot(self.logs['data_loss'][start_index:], linewidth=linewidth)
    axs[0, 1].set_title('Data loss')

    axs[1, 0].plot(self.logs['bc_in_loss'][start_index:], linewidth=linewidth)
    axs[1, 0].set_title('BC loss - Inlet')

    axs[1, 1].plot(self.logs['bc_out_loss'][start_index:], linewidth=linewidth)
    axs[1, 1].set_title('BC loss - Outlet')

    axs[2, 0].plot(self.logs['bc_down_loss'][start_index:], linewidth=linewidth)
    axs[2, 0].set_title('BC loss - Down')

    axs[2, 1].plot(self.logs['bc_up_loss'][start_index:], linewidth=linewidth)
    axs[2, 1].set_title('BC loss - Up')

    axs[3, 0].plot(self.logs['surface_loss'][start_index:], linewidth=linewidth)
    axs[3, 0].set_title('Surface loss')

    axs[3, 1].plot(self.logs['interior_loss'][start_index:], linewidth=linewidth)
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

    axs[0, 1].plot(self.lambdas['data'], linewidth=linewidth)
    axs[0, 1].set_title('lambda Data')

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
