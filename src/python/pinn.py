import torch
import torch.nn as nn
import numpy as np
import utils
import os
import pandas as pd

class PINN(nn.Module):

  def __init__(self, input_dim: int, output_dim: int, hidden_units: int, model_name: str):
    super(PINN, self).__init__()
    self.layers = nn.ModuleList()
    in_units = input_dim
    for units in hidden_units:
      layer = nn.Linear(in_units, units)
      nn.init.xavier_normal_(layer.weight)  # Apply Xavier initialization
      self.layers.append(layer)
      in_units = units
    output_layer = nn.Linear(in_units, output_dim)
    nn.init.xavier_normal_(output_layer.weight)  # Apply Xavier initialization
    self.layers.append(output_layer)
    self.logs = {"total_loss": [], 
                 "pde_ns_loss": [], "pde_ps_loss": [],
                 "bc_in_loss": [], "bc_out_loss": [], 
                  "bc_left_loss": [], "bc_right_loss": [],
                  "bc_down_loss": [], "bc_up_loss": [], 
                 "no_slip_loss": [], "real_data_loss": [], 
                 "imp_loss": []}
    self.curent_total_loss = -1
    self.current_pde_nv_loss = -1
    self.current_pde_ps_loss = -1
    self.current_bc_in_loss = -1
    self.current_bc_out_loss = -1
    self.current_bc_left_loss = -1
    self.current_bc_right_loss = -1
    self.current_bc_down_loss = -1
    self.current_bc_up_loss = -1
    self.current_no_slip_loss = -1
    self.current_real_data_loss = -1
    self.current_imp_loss = -1
    self.epoch = 0
    self.model_name = model_name


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
      z_f: torch.Tensor,
      input_b_in: torch.Tensor,
      input_b_out: torch.Tensor,
      input_b_left: torch.Tensor,
      input_b_right: torch.Tensor,
      input_b_down: torch.Tensor,
      input_b_up: torch.Tensor,
      input_s: torch.Tensor,
      input_u_points: torch.Tensor,
      output_u_exp: torch.Tensor,
      normals: torch.Tensor,
      in_velocity: torch.Tensor,
      mu: float, rho: float, 
      c1: float, c2: float, c3: float, c4: float, c5: float, c6: float, c7: float, c8: float, c9: float, c10: float, c11:float
  ) -> torch.Tensor:

    input_f = utils.stack_xyz_tensors(x_f, y_f, z_f)

    output_f = self(input_f)
    u = output_f[:, 0]
    v = output_f[:, 1]
    w = output_f[:, 2]
    p = output_f[:, 3]

    u_x = utils.grad(u, x_f)
    u_y = utils.grad(u, y_f)
    u_z = utils.grad(u, z_f)
    u_xx = utils.grad(u_x, x_f, False)
    u_yy = utils.grad(u_y, y_f, False)
    u_zz = utils.grad(u_z, z_f, False)

    v_x = utils.grad(v, x_f)
    v_y = utils.grad(v, y_f)
    v_z = utils.grad(v, z_f)
    v_xx = utils.grad(v_x, x_f, False)
    v_yy = utils.grad(v_y, y_f, False)
    v_zz = utils.grad(v_z, z_f, False)

    w_x = utils.grad(w, x_f)
    w_y = utils.grad(w, y_f)
    w_z = utils.grad(w, z_f)
    w_xx = utils.grad(w_x, x_f, False)
    w_yy = utils.grad(w_y, y_f, False)
    w_zz = utils.grad(w_z, z_f, False)

    p_x = utils.grad(p, x_f)
    p_xx = utils.grad(p_x, x_f, False)
    p_y = utils.grad(p, y_f)
    p_yy = utils.grad(p_y, y_f, False)
    p_z = utils.grad(p, z_f)
    p_zz = utils.grad(p_z, z_f, False)

    b = self.__compute_b(u=u, v=v, w=w, 
                       u_x=u_x, u_y=u_y, u_z=u_z, 
                       v_x=v_x, v_y=v_y, v_z=v_z, 
                       w_x=w_x, w_y=w_y, w_z=w_z, 
                       rho=rho)

    # PDE loss
    ## Navier-Stokes equations
    ### X-momentum equation
    f1 = u*u_x + v*u_y + w*u_z + (1/rho) * p_x - mu * (u_xx + u_yy + u_zz)
    ### Y-momentum equation
    f2 = u*v_x + v*v_y + w*v_z + (1/rho) * p_y - mu * (v_xx + v_yy + v_zz)
    ### Z-momentum equation
    f3 = u*w_x + v*w_y + w*w_z + (1/rho) * p_z - mu * (w_xx + w_yy + w_zz)
    ### Continuity equation
    f4 = u_x + v_y + w_z
    ### Poisson equation
    f5 = p_xx + p_yy + p_zz - b

    pde_ns_loss =  100 * (1/4) * torch.mean(torch.square(f1)) + \
                        torch.mean(torch.square(f2)) + \
                        torch.mean(torch.square(f3)) + \
                        torch.mean(torch.square(f4))
    
    pde_ps_loss = 100 * torch.mean(torch.square(f5))
    
    # Boundary conditions loss
    ## Inlet: u = 0, v = -in_velocity, w = 0 & p = 1 for y = y_max
    output_b_in = self(input_b_in)
    u_b_in_pred = output_b_in[:, 0]
    v_b_in_pred = output_b_in[:, 1]
    w_b_in_pred = output_b_in[:, 2]
    v_b_in_true = torch.full_like(v_b_in_pred, -1 * in_velocity)
    bc_in_loss_u = torch.mean(torch.square(u_b_in_pred))
    bc_in_loss_v = torch.mean(torch.square(v_b_in_pred - v_b_in_true))
    bc_in_loss_w = torch.mean(torch.square(w_b_in_pred))
    bc_in_loss = 100 * (1/3) * (bc_in_loss_u + bc_in_loss_v + bc_in_loss_w)

    ## Outlet: p = 1 for y = 0
    output_b_out = self(input_b_out)
    p_b_out_pred = output_b_out[:, 3]
    p_b_out_true = torch.ones_like(p_b_out_pred)
    bc_out_loss_p = torch.mean(torch.square(p_b_out_pred - p_b_out_true))
    bc_out_loss = 100 * bc_out_loss_p

    ## Left (Far-field): p = 1 for x = 0
    output_b_left = self(input_b_left)
    p_b_left_pred = output_b_left[:, 3]
    p_b_left_true = torch.ones_like(p_b_left_pred)
    bc_left_loss_p = torch.mean(torch.square(p_b_left_pred - p_b_left_true))
    bc_left_loss = 100 * bc_left_loss_p

    ## Right (Far-field): p = 1 for x = x_max
    output_b_right = self(input_b_right)
    p_b_right_pred = output_b_right[:, 3]
    p_b_right_true = torch.ones_like(p_b_right_pred)
    bc_right_loss_p = torch.mean(torch.square(p_b_right_pred - p_b_right_true))
    bc_right_loss = 100 * bc_right_loss_p

    # Down (No-slip wall): u = 0, v = 0, w = 0 for z = 0
    output_b_down = self(input_b_down)
    u_b_down_pred = output_b_down[:, 0]
    v_b_down_pred = output_b_down[:, 1]
    w_b_down_pred = output_b_down[:, 2]
    bc_down_loss_u = torch.mean(torch.square(u_b_down_pred))
    bc_down_loss_v = torch.mean(torch.square(v_b_down_pred))
    bc_down_loss_w = torch.mean(torch.square(w_b_down_pred))
    bc_down_loss = 100 * (1/3) * (bc_down_loss_u + bc_down_loss_v + bc_down_loss_w)

    # Up (Far-field Conditions): p = 1 for z = z_max
    output_b_up = self(input_b_up)
    p_b_up_pred = output_b_up[:, 3]
    p_b_up_true = torch.ones_like(p_b_up_pred)
    bc_up_loss_p = torch.mean(torch.square(p_b_up_pred - p_b_up_true))
    bc_up_loss = 100 * bc_up_loss_p

    # Object surface boundary conditions loss
    output_s = self(input_s)
    u_s_pred = output_s[:, 0]
    v_s_pred = output_s[:, 1]
    w_s_pred = output_s[:, 2]

    ## no-slip condition
    no_slip_loss_u = torch.mean(torch.square(u_s_pred))
    no_slip_loss_v = torch.mean(torch.square(v_s_pred))
    no_slip_loss_w = torch.mean(torch.square(w_s_pred))
    no_slip_loss = 100 * (1/3) * (no_slip_loss_u + no_slip_loss_v + no_slip_loss_w)

    # Real measurements loss
    # output_u = self(input_u)
    # u_u_pred = output_u[:, 0]
    # v_u_pred = output_u[:, 1]
    # w_u_pred = output_u[:, 2]
    # u_u_exp = output_u_exp[:, 0]
    # v_u_exp = output_u_exp[:, 1]
    # w_u_exp = output_u_exp[:, 2]
    # real_data_loss_u = torch.mean(torch.square(u_u_pred - u_u_exp))
    # real_data_loss_v = torch.mean(torch.square(v_u_pred - v_u_exp))
    # real_data_loss_w = torch.mean(torch.square(w_u_pred - w_u_exp))
    # real_data_loss = 100 * (1/3) * (real_data_loss_u + real_data_loss_v + real_data_loss_w)
    real_data_loss = torch.tensor(0)

    # Impermeability condition loss
    # dot_products = torch.sum(output_s[:, :3] * normals, dim=1)
    # imp_loss = 100 * torch.mean(torch.square(dot_products))
    imp_loss = torch.tensor(0)

    # total loss
    total_loss =  c1 * pde_ns_loss + \
                  c2 * pde_ps_loss + \
                  c3 * bc_in_loss + \
                  c4 * bc_out_loss + \
                  c5 * bc_left_loss + \
                  c6 * bc_right_loss + \
                  c7 * bc_down_loss + \
                  c8 * bc_up_loss + \
                  c9 * no_slip_loss + \
                  c10 * real_data_loss + \
                  c11 * imp_loss

    return total_loss, pde_ns_loss, pde_ps_loss, \
           bc_in_loss, bc_out_loss, \
           bc_left_loss, bc_right_loss, \
           bc_down_loss, bc_up_loss, \
           no_slip_loss, \
           real_data_loss, \
            imp_loss


  def closure(
      self, 
      s_df: pd.DataFrame, n_df: pd.DataFrame, u_df: pd.DataFrame,
      optimizer: torch.optim.Optimizer, 
      Nf: int, Nb: int, Ns: int, Nu: int, 
      x_max: float, y_max: float, z_max: float, 
      c1: float, c2: float, c3: float, c4: float, c5: float, c6: float, c7: float, c8: float, c9: float, c10: float, c11: float,
      in_velocity: int, 
      mu: float, rho: float, 
      device: torch.device
  ) -> torch.Tensor:

    optimizer.zero_grad()

    training_input = self.__generate_inputs(s_df, n_df, u_df, x_max, y_max, z_max, Nf, Nb, Ns, Nu, device)

    total_loss, pde_ns_loss, pde_ps_loss, bc_in_loss, bc_out_loss, bc_left_loss, bc_right_loss, bc_down_loss, bc_up_loss, no_slip_loss, real_data_loss, imp_loss = self.loss(
                    *training_input,
                    in_velocity,
                    mu, rho,
                    c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, c7=c7, c8=c8, c9=c9, c10=c10, c11=c11)

    self.current_total_loss = total_loss.item()
    self.current_pde_ns_loss = pde_ns_loss.item()
    self.current_pde_ps_loss = pde_ps_loss.item()
    self.current_bc_in_loss = bc_in_loss.item()
    self.current_bc_out_loss = bc_out_loss.item()
    self.current_bc_left_loss = bc_left_loss.item()
    self.current_bc_right_loss = bc_right_loss.item()
    self.current_bc_down_loss = bc_down_loss.item()
    self.current_bc_up_loss = bc_up_loss.item()
    self.current_no_slip_loss = no_slip_loss.item()
    self.current_real_data_loss = real_data_loss.item()
    self.current_imp_loss = imp_loss.item()

    total_loss.backward()

    return total_loss


  def eval_pinn(
      self, 
      s_df: pd.DataFrame, n_df: pd.DataFrame, u_df: pd.DataFrame, 
      Nf: int, N0: int, Nb: int, Ns: int, Nu: int, 
      x_max: float, y_max: float, z_max: float, 
      in_velocity: int, 
      mu: float, rho: float, 
      device: torch.device,
      c1 = 1., c2 = 1., c3 = 1., c4 = 1., c5 = 1., c6 = 1., c7 = 1., c8 = 1., c9 = 1., c10=1., c11=1.) -> torch.Tensor:

    Nf = utils.nearest_power_of_2(Nf)
    Nb = utils.nearest_power_of_2(Nb)
    Ns = utils.nearest_power_of_2(Ns)

    training_input = self.__generate_inputs(s_df, u_df, x_max, y_max, z_max, Nf, Nb, Ns, Nu, device)

    return [_loss.item() for _loss in self.loss(*training_input,
                                                in_velocity,
                                                mu, rho,
                                                c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, c7=c7, c8=c8, c9=c9, c10=c10, c11=c11)]


  def train_pinn(
        self, 
        epochs: int, 
        optimizer: torch.optim.Optimizer, 
        s_df: pd.DataFrame,
        n_df: pd.DataFrame,
        u_df: pd.DataFrame,
        Nf: int, Nb: int, Ns: int, Nu: int,
        x_max: float, y_max: float, z_max: float,
        in_velocity: int,
        mu: float, rho: float,
        device: torch.device,
        checkpoint_epochs: int,
        model_dir: str,
        c1 = 1., c2 = 1., c3 = 1., c4 = 1., c5 = 1., c6 = 1., c7 = 1., c8 = 1., c9 = 1., c10=1., c11=1.):

    print("=======================================================")
    print(self)
    print(f"Model name: {self.model_name}")
    print(f"Number of epochs: {epochs}")
    print(f"Number of collocation points Nf: {Nf}")
    print(f"Number of boundary condition points Nb: {Nb}")
    print(f"Number of object surface points Ns: {Ns}")
    print(f"Number of real data points Nu: {Nu}")
    print(f"X max: {x_max}, Y max: {y_max}, Z max: {z_max}")
    print(f"mu: {mu}, rho: {rho}")
    print(f"c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}, c5: {c5}, c6: {c6}, c7: {c7}, c8: {c8}, c9: {c9}, c10: {c10}")
    print(f"Inflow velocity: {in_velocity}")
    print(f"Device: {device}")
    print(f"Checkpoint epochs: {checkpoint_epochs}")
    print(f"Model directory: {model_dir}")
    print("=======================================================")
    print("=> converting Nf, Nb, Ns, Nu to nearest power of 2...") # Quasi - Monte Carlo sampling requirement

    Nf = utils.nearest_power_of_2(Nf)
    Nb = utils.nearest_power_of_2(Nb)
    Ns = utils.nearest_power_of_2(Ns)
    Nu = utils.nearest_power_of_2(Nu)

    print(f"Nf: {Nf}, Nb: {Nb}, Ns: {Ns}, Nu: {Nu}")
    print("=======================================================")
    print("=> starting training...")
    print("=======================================================")

    try:
      while self.epoch < epochs:

        self.epoch += 1

        optimizer.step(lambda: 
                      self.closure(
                        s_df=s_df, 
                        n_df=n_df, 
                        u_df=u_df, 
                        optimizer=optimizer, 
                        Nf=Nf, Nb=Nb, Ns=Ns, Nu=Nu,
                        x_max=x_max, y_max=y_max, z_max=z_max,
                        c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, c7=c7, c8=c8, c9=c9, c10=c10, c11=c11,
                        in_velocity=in_velocity,
                        mu=mu, rho=rho,
                        device=device))

        self.__log_metrics(self.current_total_loss, 
                           self.current_pde_ns_loss, self.current_pde_ps_loss, 
                           self.current_bc_in_loss, self.current_bc_out_loss, 
                           self.current_bc_left_loss, self.current_bc_right_loss, 
                           self.current_bc_down_loss, self.current_bc_up_loss, 
                           self.current_no_slip_loss, self.current_real_data_loss, 
                           self.current_imp_loss)
        self.print_current_metrics() 
        
        if np.isnan(self.current_total_loss):
          print("=> NaN loss...")
          self, optimizer = self.load_last_checkpoint(optimizer, model_dir)
          continue

        if self.epoch % checkpoint_epochs == 0:
          checkpoint_path = os.path.join(model_dir, self.model_name, str(self.epoch) + ".pt")
          self.__save_checkpoint(optimizer, checkpoint_path)
          
    except KeyboardInterrupt:
      print(f"Training stopped by user at epoch {self.epoch}.")
      self.epoch -= 1


  def __compute_b(
        self, 
        u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, 
        u_x: torch.Tensor, u_y: torch.Tensor, u_z: torch.Tensor, 
        v_x: torch.Tensor, v_y: torch.Tensor, v_z: torch.Tensor, 
        w_x: torch.Tensor, w_y: torch.Tensor, w_z: torch.Tensor, 
        rho: torch.Tensor) -> torch.Tensor:
    # u, v, w: velocity components
    # u_t, v_t, w_t: time derivatives of the velocity components
    # u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z: spatial derivatives of the velocity components
    # rho: fluid density (either a constant or an array)

    # Calculate the divergence of the velocity field
    # div_u = u_x + v_y + w_z

    # Convective acceleration term (tensor product of velocity gradient with its transpose)
    convective_acc = u * u_x + v * u_y + w * u_z + u * v_x + v * v_y + w * v_z + u * w_x + v * w_y + w * w_z
    # convective_acc = u * u_x + v * v_y + w * w_z

    return rho * convective_acc


  def __generate_inputs(
        self, 
        s_df: pd.DataFrame, n_df: pd.DataFrame, u_df: pd.DataFrame, 
        x_max: float, y_max: float, z_max: float,  
        Nf: int, Nb: int, Ns: int, Nu: int, 
        device: torch.device) -> tuple:

    # collocation points
    samples_f = utils.qmc_sample_points_in_domain_3d(_x_min=0, _x_max=x_max, 
                                                     _y_min=0, _y_max=y_max, 
                                                     _z_min=0, _z_max=y_max, 
                                                     num_samples=Nf)

    x_f = utils.tensor_from_array(samples_f[0], device=device, requires_grad=True)
    y_f = utils.tensor_from_array(samples_f[1], device=device, requires_grad=True)
    z_f = utils.tensor_from_array(samples_f[2], device=device, requires_grad=True)

    # boundary condition points
    ## inflow, y=1
    Nb_in = utils.nearest_power_of_2(int(Nb/6))
    samples_b_in = utils.qmc_sample_points_in_domain_2d(_x_min=0, _x_max=x_max,
                                                        _y_min=0, _y_max=z_max,
                                                        num_samples=Nb_in)

    x_b_in = utils.tensor_from_array(samples_b_in[0], device=device, requires_grad=False)
    y_b_in = utils.tensor_from_array(utils.ones(Nb_in), device=device, requires_grad=False)
    z_b_in = utils.tensor_from_array(samples_b_in[1], device=device, requires_grad=False)
    xyz_b_in = utils.stack_xyz_tensors(x_b_in, y_b_in, z_b_in)

    ## outflow, y=0
    Nb_out = utils.nearest_power_of_2(int(Nb/6))
    samples_b_out = utils.qmc_sample_points_in_domain_2d(_x_min=0, _x_max=x_max,
                                                         _y_min=0, _y_max=z_max,
                                                         num_samples=Nb_out)

    x_b_out = utils.tensor_from_array(samples_b_out[0], device=device, requires_grad=False)
    y_b_out = utils.tensor_from_array(utils.zeros(Nb_out), device=device, requires_grad=False)
    z_b_out = utils.tensor_from_array(samples_b_out[1], device=device, requires_grad=False)
    xyz_b_out = utils.stack_xyz_tensors(x_b_out, y_b_out, z_b_out)

    ## left, x=0
    Nb_left = utils.nearest_power_of_2(int(Nb/6))
    samples_b_left = utils.qmc_sample_points_in_domain_2d(_x_min=0, _x_max=y_max,
                                                          _y_min=0, _y_max=z_max,
                                                          num_samples=Nb_left)

    x_b_left = utils.tensor_from_array(utils.zeros(Nb_left), device=device, requires_grad=False)
    y_b_left = utils.tensor_from_array(samples_b_left[0], device=device, requires_grad=False)
    z_b_left = utils.tensor_from_array(samples_b_left[1], device=device, requires_grad=False)
    xyz_b_left = utils.stack_xyz_tensors(x_b_left, y_b_left, z_b_left)

    ## right, x=1
    Nb_right = utils.nearest_power_of_2(int(Nb/6))
    samples_b_right = utils.qmc_sample_points_in_domain_2d(_x_min=0, _x_max=y_max,
                                                           _y_min=0, _y_max=z_max,
                                                           num_samples=Nb_right)

    x_b_right = utils.tensor_from_array(utils.ones(Nb_right), device=device, requires_grad=False)
    y_b_right = utils.tensor_from_array(samples_b_right[0], device=device, requires_grad=False)
    z_b_right = utils.tensor_from_array(samples_b_right[1], device=device, requires_grad=False)
    xyz_b_right = utils.stack_xyz_tensors(x_b_right, y_b_right, z_b_right)

    ## down, z=0
    Nb_down = utils.nearest_power_of_2(int(Nb/6))
    samples_b_down = utils.qmc_sample_points_in_domain_2d(_x_min=0, _x_max=x_max,
                                                          _y_min=0, _y_max=y_max,
                                                          num_samples=Nb_down)

    x_b_down = utils.tensor_from_array(samples_b_down[0], device=device, requires_grad=False)
    y_b_down = utils.tensor_from_array(samples_b_down[1], device=device, requires_grad=False)
    z_b_down = utils.tensor_from_array(utils.zeros(Nb_down), device=device, requires_grad=False)
    xyz_b_down = utils.stack_xyz_tensors(x_b_down, y_b_down, z_b_down)

    ## up, z=1
    Nb_up = utils.nearest_power_of_2(int(Nb/6))
    samples_b_up = utils.qmc_sample_points_in_domain_2d(_x_min=0, _x_max=x_max,
                                                        _y_min=0, _y_max=y_max,
                                                        num_samples=Nb_up)

    x_b_up = utils.tensor_from_array(samples_b_up[0], device=device, requires_grad=False)
    y_b_up = utils.tensor_from_array(samples_b_up[1], device=device, requires_grad=False)
    z_b_up = utils.tensor_from_array(utils.ones(Nb_up), device=device, requires_grad=False)
    xyz_b_up = utils.stack_xyz_tensors(x_b_up, y_b_up, z_b_up)

    # points & normal vectors on the surface of the object
    ## sample Ns object surface points with the corresponding normals
    sampled_indices_s = s_df.sample(n=Ns).index

    x_s, y_s, z_s = [utils.tensor_from_array(s_df.loc[sampled_indices_s, col].values, device=device, requires_grad=False) for col in ['x', 'y', 'z']]
    n_x, n_y, n_z = [utils.tensor_from_array(n_df.loc[sampled_indices_s, col].values, device=device, requires_grad=False) for col in ['x', 'y', 'z']]

    xyz_s = utils.stack_xyz_tensors(x_s, y_s, z_s)
    n_xyz = utils.stack_xyz_tensors(n_x, n_y, n_z)

    # points & velocity of the real measurements
    ## sample Nu points with the corresponding measurements
    sampled_indices_u = u_df.sample(n=Nu).index

    x_u, y_u, z_u, u_u, v_u, w_u = [utils.tensor_from_array(u_df.loc[sampled_indices_u, col].values, device=device, requires_grad=False) for col in ['x', 'y', 'z', 'u', 'v', 'w']]
    xyz_u = utils.stack_xyz_tensors(x_u, y_u, z_u)
    uyw_u = utils.stack_xyz_tensors(u_u, v_u, w_u)

    return (x_f, y_f, z_f, xyz_b_in, xyz_b_out, xyz_b_left, xyz_b_right, xyz_b_down, xyz_b_up, xyz_s, xyz_u, uyw_u, n_xyz)


  def __log_metrics(self, total_loss: float, 
                    pde_ns_loss: float, pde_ps_loss: float, 
                    bc_in_loss: float, bc_out_loss: float, 
                    bc_left_loss: float, bc_right_loss: float, 
                    bc_down_loss: float, bc_up_loss: float, 
                    no_slip_loss: float, real_data_loss: float,
                    imp_loss: float):
    """ Log training metrics """
    self.logs['total_loss'].append(total_loss)
    self.logs['pde_ns_loss'].append(pde_ns_loss)
    self.logs['pde_ps_loss'].append(pde_ps_loss)
    self.logs['bc_in_loss'].append(bc_in_loss)
    self.logs['bc_out_loss'].append(bc_out_loss)
    self.logs['bc_left_loss'].append(bc_left_loss)
    self.logs['bc_right_loss'].append(bc_right_loss)
    self.logs['bc_down_loss'].append(bc_down_loss)
    self.logs['bc_up_loss'].append(bc_up_loss)
    self.logs['no_slip_loss'].append(no_slip_loss)
    self.logs['real_data_loss'].append(real_data_loss)
    self.logs['imp_loss'].append(imp_loss)


  def __get_logs(self):
    """ Retrieve the logged metrics """
    return self.logs


  def print_current_metrics(self):
      """ Print the most recent set of metrics """
      if self.logs['total_loss']:
          print(f"Epoch: {self.epoch}, "
                f"Total Loss: {self.logs['total_loss'][-1]:.4f}, "
                f"PDE Loss - Navier Stoker: {self.logs['pde_ns_loss'][-1]:.4f}, "
                f"PDE Loss - Poisson: {self.logs['pde_ps_loss'][-1]:.4f}, "
                f"BC Inlet Loss: {self.logs['bc_in_loss'][- 1]:.4f}, "
                f"BC Outlet Loss: {self.logs['bc_out_loss'][- 1]:.4f}, "
                f"BC Left Loss: {self.logs['bc_left_loss'][- 1]:.4f}, "
                f"BC Right Loss: {self.logs['bc_right_loss'][- 1]:.4f}, "
                f"BC Down Loss: {self.logs['bc_down_loss'][- 1]:.4f}, "
                f"BC Up Loss: {self.logs['bc_up_loss'][- 1]:.4f}, "
                f"No-Slip Loss: {self.logs['no_slip_loss'][-1]:.4f}, "
                f"Real-Data Loss: {self.logs['real_data_loss'][-1]:.4f}, " 
                f"Impermeability Loss: {self.logs['imp_loss'][-1]:.4f}")
      else:
          print("No metrics to display.")


  def print_all_metrics(self):
      """ Print all metrics """

      for _epoch in range(1, self.epoch + 1): 
      
        if self.logs['total_loss']:
            print(f"Epoch: {_epoch}, "
                  f"Total Loss: {self.logs['total_loss'][_epoch - 1]:.4f}, "
                  f"PDE Loss - Navier Stoker: {self.logs['pde_ns_loss'][_epoch - 1]:.4f}, "
                  f"PDE Loss - Poisson: {self.logs['pde_ps_loss'][_epoch - 1]:.4f}, "
                  f"BC Inlet Loss: {self.logs['bc_in_loss'][_epoch - 1]:.4f}, "
                  f"BC Outlet Loss: {self.logs['bc_out_loss'][_epoch - 1]:.4f}, "
                  f"BC Left Loss: {self.logs['bc_left_loss'][_epoch - 1]:.4f}, "
                  f"BC Right Loss: {self.logs['bc_right_loss'][_epoch - 1]:.4f}, "
                  f"BC Down Loss: {self.logs['bc_down_loss'][_epoch - 1]:.4f}, "
                  f"BC Up Loss: {self.logs['bc_up_loss'][_epoch - 1]:.4f}, "
                  f"No-Slip Loss: {self.logs['no_slip_loss'][_epoch - 1]:.4f}, "
                  f"Real-Data Loss: {self.logs['real_data_loss'][_epoch - 1]:.4f}, "
                  f"Impermeability Loss: {self.logs['imp_loss'][_epoch - 1]:.4f}")
        else:
            print("No metrics to display.")


  def __save_checkpoint(self, optimizer: torch.optim.Optimizer, file_path: str):

    print("=> saving checkpoint '{}'".format(file_path))
    state = {'name': self.model_name, 'epoch': self.epoch, 'state_dict': self.state_dict(),
              'optimizer': optimizer.state_dict(), "logs": self.logs}
    torch.save(state, file_path)


  def __load_checkpoint(self, optimizer: torch.optim.Optimizer, file_path: str, mode='training') -> 'PINN, torch.optim.Optimizer':

      if os.path.isfile(file_path):
          print("=> loading checkpoint '{}'".format(file_path))
          checkpoint = torch.load(file_path)
          self.load_state_dict(checkpoint['state_dict'])
          self.model_name = checkpoint['name']
          self.epoch = checkpoint['epoch']
          self.logs = checkpoint['logs']
          
          if mode == 'training':
            optimizer.load_state_dict(checkpoint['optimizer'])

          print("=> loaded checkpoint '{}' (epoch {})"
                    .format(file_path, checkpoint['epoch']))
      else:
          print("=> no checkpoint found at '{}'".format(file_path))

      return self, optimizer


  def load_last_checkpoint(self, optimizer: torch.optim.Optimizer, checkpoint_dir: str) -> 'PINN, torch.optim.Optimizer':

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


  def load_checkpoint_num(self, optimizer: torch.optim.Optimizer, checkpoint_dir, model_name: str, checkpoint_num: int, mode='training') -> 'PINN, torch.optim.Optimizer':

    file_path = os.path.join(checkpoint_dir, model_name, str(checkpoint_num) + ".pt")
    return self.__load_checkpoint(optimizer, file_path, mode)