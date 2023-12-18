import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
import utils

class PINN(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_units):
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
    self.logs = {"total_loss": [], "pde_loss": [], "ic_loss": [], "bc_loss": [], "no_slip_loss": []}
    self.curent_total_loss = -1
    self.current_pde_loss = -1
    self.current_ic_loss = -1
    self.current_bc_loss = -1
    self.current_no_slip_loss = -1
    self.epoch = 0

  def forward(self, input):
    for layer in self.layers[:-1]:
      # output = torch.sigmoid(layer(input))
      output = torch.tanh(layer(input))
      # output = torch.relu(layer(input))
      input = output
    output = self.layers[-1](input)
    return output

  def grad(self, x, y, create_graph=True):
    return torch.autograd.grad(x, y, grad_outputs=torch.ones_like(x), create_graph=create_graph, retain_graph=True, only_inputs=True)[0]

  def loss(self, 
        x_f,
        y_f,
        z_f,
        t_f,
        input_0,
        input_b,
        input_w,
        in_velocity,
        mu, rho, c1, c2, c3, c4):

    input_f = utils.stack_xyzt_tensors(x_f, y_f, z_f, t_f)

    output_f = self(input_f)
    u = output_f[:, 0]
    v = output_f[:, 1]
    w = output_f[:, 2]
    p = output_f[:, 3]

    u_t = self.grad(u, t_f)
    u_x = self.grad(u, x_f)
    u_y = self.grad(u, y_f)
    u_z = self.grad(u, z_f)
    u_xx = self.grad(u_x, x_f, False)
    u_yy = self.grad(u_y, y_f, False)
    u_zz = self.grad(u_z, z_f, False)

    v_t = self.grad(v, t_f)
    v_x = self.grad(v, x_f)
    v_y = self.grad(v, y_f)
    v_z = self.grad(v, z_f)
    v_xx = self.grad(v_x, x_f, False)
    v_yy = self.grad(v_y, y_f, False)
    v_zz = self.grad(v_z, z_f, False)

    w_t = self.grad(w, t_f)
    w_x = self.grad(w, x_f)
    w_y = self.grad(w, y_f)
    w_z = self.grad(w, z_f)
    w_xx = self.grad(w_x, x_f, False)
    w_yy = self.grad(w_y, y_f, False)
    w_zz = self.grad(w_z, z_f, False)

    p_x = self.grad(p, x_f)
    p_xx = self.grad(p_x, x_f, False)
    p_y = self.grad(p, y_f)
    p_yy = self.grad(p_y, y_f, False)
    p_z = self.grad(p, z_f)
    p_zz = self.grad(p_z, z_f, False)

    b = self.compute_b(u=u, v=v, w=w, t=t_f, 
                       u_x=u_x, u_y=u_y, u_z=u_z, 
                       v_x=v_x, v_y=v_y, v_z=v_z, 
                       w_x=w_x, w_y=w_y, w_z=w_z, 
                       rho=rho)

    # PDE loss
    ## Navier-Stokes equations
    ### X-momentum equation
    f1 = u_t + u*u_x + v*u_y + w*u_z + (1/rho) * p_x - mu * (u_xx + u_yy + u_zz)
    ### Y-momentum equation
    f2 = v_t + u*v_x + v*v_y + w*v_z + (1/rho) * p_y - mu * (v_xx + v_yy + v_zz)
    ### Z-momentum equation
    f3 = w_t + u*w_x + v*w_y + w*w_z + (1/rho) * p_z - mu * (w_xx + w_yy + w_zz)
    ### Continuity equation
    f4 = u_x + v_y + w_z
    ### Poisson equation
    f5 = p_xx + p_yy + p_zz - b

    pde_loss =  100 * (1/5) * torch.mean(torch.square(f1)) + \
                        torch.mean(torch.square(f2)) + \
                        torch.mean(torch.square(f3)) + \
                        torch.mean(torch.square(f4)) + \
                        torch.mean(torch.square(f5))

    # Initial condition loss
    output_0 = self(input_0)
    u0_pred = output_0[:, 0]
    v0_pred = output_0[:, 1]
    w0_pred = output_0[:, 2]
    p0_pred = output_0[:, 3]

    # for t = 0 -> u, v, w = 0, p = 1
    p0_true = torch.ones_like(p0_pred)

    ic_loss_u = torch.mean(torch.square(u0_pred))
    ic_loss_v = torch.mean(torch.square(v0_pred))
    ic_loss_w = torch.mean(torch.square(w0_pred))
    ic_loss_p = torch.mean(torch.square(p0_pred - p0_true))

    ic_loss = 100 * (1/4) * (ic_loss_u + ic_loss_v + ic_loss_w + ic_loss_p)

    # Boundary conditions loss
    output_b = self(input_b)
    u_b_pred = output_b[:, 0]
    v_b_pred = output_b[:, 1]
    w_b_pred = output_b[:, 2]

    # u = 0, v = -1 * in_velocity and w = 0 for y = 1
    v_b_true = torch.full_like(v_b_pred, -1 * in_velocity)
    
    bc_loss_u = torch.mean(torch.square(u_b_pred))
    bc_loss_v = torch.mean(torch.square(v_b_pred - v_b_true))
    bc_loss_w = torch.mean(torch.square(w_b_pred))

    bc_loss = 100 * (1/3) * (bc_loss_u + bc_loss_v + bc_loss_w)

    # Wing surface boundary conditions loss
    output_wing = self(input_w)
    u_w_pred = output_wing[:, 0]
    v_w_pred = output_wing[:, 1]
    w_w_pred = output_wing[:, 2]

    ## no-slip condition
    no_slip_loss_u = torch.mean(torch.square(u_w_pred))
    no_slip_loss_v = torch.mean(torch.square(v_w_pred))
    no_slip_loss_w = torch.mean(torch.square(w_w_pred))

    no_slip_loss = 100 * (1/3) * (no_slip_loss_u + no_slip_loss_v + no_slip_loss_w)

    # total loss
    total_loss =  c1 * pde_loss + \
                  c2 * ic_loss + \
                  c3 * bc_loss + \
                  c4 * no_slip_loss

    return total_loss, pde_loss, ic_loss, bc_loss, no_slip_loss
  
  def compute_b(self, u, v, w, t, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z, rho):
    # u, v, w: velocity components
    # u_t, v_t, w_t: time derivatives of the velocity components
    # u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z: spatial derivatives of the velocity components
    # rho: fluid density (either a constant or an array)

    # Calculate the divergence of the velocity field
    div_u = u_x + v_y + w_z

    # Time derivative of the divergence of the velocity field
    div_u_t = self.grad(div_u, t, create_graph=False)

    # Convective acceleration term (tensor product of velocity gradient with its transpose)
    convective_acc = (u * u_x + v * u_y + w * u_z +
                      u * v_x + v * v_y + w * v_z +
                      u * w_x + v * w_y + w * w_z)
    
    return rho * (div_u_t + convective_acc)
  
  def closure(self, 
              wing_df,
              optimizer, 
              Nf, N0, Nb, Nw, 
              x_max, y_max, z_max, t_max, 
              c1, c2, c3, c4, 
              in_velocity, 
              mu, rho, 
              device):

    optimizer.zero_grad()

    training_input = self.create_training_inputs(wing_df, x_max, y_max, z_max, t_max, Nf, N0, Nb, Nw, device)

    total_loss, pde_loss, ic_loss, bc_loss, no_slip_loss = self.loss(
                    *training_input,
                    in_velocity,
                    mu, rho,
                    c1=c1, c2=c2, c3=c3, c4=c4)

    self.current_total_loss = total_loss.item()
    self.current_pde_loss = pde_loss.item()
    self.current_ic_loss = ic_loss.item()
    self.current_bc_loss = bc_loss.item()
    self.current_no_slip_loss = no_slip_loss.item()

    total_loss.backward()

    return total_loss

  def create_training_inputs(self, wing_df, x_max, y_max, z_max, t_max, Nf, N0, Nb, Nw, device):
    # TODO: use quasi monte carlo sampling
    # collocation points
    x_f = utils.tensor_from_array(utils.sample_points_in_domain(0, x_max, Nf), device=device, requires_grad=True)
    y_f = utils.tensor_from_array(utils.sample_points_in_domain(0, y_max, Nf), device=device, requires_grad=True)
    z_f = utils.tensor_from_array(utils.sample_points_in_domain(0, z_max, Nf), device=device, requires_grad=True)
    t_f = utils.tensor_from_array(utils.sample_points_in_domain(0, t_max, Nf), device=device, requires_grad=True)
    # xyzt_f = utils.stack_xyzt_tensors(x_f, y_f, z_f, t_f)
    # if stacked in a single tensor, the gradients are not computed correctly

    # initial condition points (t=0)
    x0 = utils.tensor_from_array(utils.sample_points_in_domain(0, x_max, N0), device=device, requires_grad=False)
    y0 = utils.tensor_from_array(utils.sample_points_in_domain(0, y_max, N0), device=device, requires_grad=False)
    z0 = utils.tensor_from_array(utils.sample_points_in_domain(0, z_max, N0), device=device, requires_grad=False)
    t0 = utils.tensor_from_array(utils.zeros(N0), device=device, requires_grad=False)
    xyzt_0 = utils.stack_xyzt_tensors(x0, y0, z0, t0)

    # boundary condition points (inflow, y=1)
    x_b = utils.tensor_from_array(utils.sample_points_in_domain(0, x_max, Nb), device=device, requires_grad=False)
    y_b = utils.tensor_from_array(utils.ones(Nb), device=device, requires_grad=False)
    z_b = utils.tensor_from_array(utils.sample_points_in_domain(0, z_max, Nb), device=device, requires_grad=False)
    t_b = utils.tensor_from_array(utils.sample_points_in_domain(0, t_max, Nb), device=device, requires_grad=False)
    xyzt_b = utils.stack_xyzt_tensors(x_b, y_b, z_b, t_b)

    # points & normal vectors on the surface of the wing
    ## sample Nw wing points with the corresponding normals
    sampled_indices = wing_df.sample(n=Nw).index

    x_w, y_w, z_w = [utils.tensor_from_array(wing_df.loc[sampled_indices, col].values, device=device, requires_grad=False) for col in ['x', 'y', 'z']]
    # n_x, n_y, n_z = [utils.tensor_from_array(norm_df.loc[sampled_indices, col].values, device=device, requires_grad=False) for col in ['x', 'y', 'z']]
    t_w = utils.tensor_from_array(utils.sample_points_in_domain(0, t_max, Nw), device=device, requires_grad=False)

    xyzt_w = utils.stack_xyzt_tensors(x_w, y_w, z_w, t_w)
    # n_xyz = utils.stack_xyz_tensors(n_x, n_y, n_z)

    return (x_f, y_f, z_f, t_f, xyzt_0, xyzt_b, xyzt_w)

  def log_metrics(self, total_loss, pde_loss, ic_loss, bc_loss, no_slip_loss):
    """ Log training metrics """
    self.logs['total_loss'].append(total_loss)
    self.logs['pde_loss'].append(pde_loss)
    self.logs['ic_loss'].append(ic_loss)
    self.logs['bc_loss'].append(bc_loss)
    self.logs['no_slip_loss'].append(no_slip_loss)

  def get_logs(self):
    """ Retrieve the logged metrics """
    return self.logs

  def print_current_metrics(self):
      """ Print the most recent set of metrics """
      if self.logs['total_loss']:
          print(f"Epoch: {self.epoch}, "
                f"Total Loss: {self.logs['total_loss'][-1]:.4f}, "
                f"PDE Loss: {self.logs['pde_loss'][-1]:.4f}, "
                f"IC Loss: {self.logs['ic_loss'][-1]:.4f}, "
                f"BC Loss: {self.logs['bc_loss'][-1]:.4f}, "
                f"No Slip Loss: {self.logs['no_slip_loss'][-1]:.4f}")
      else:
          print("No metrics to display.")

  def print_all_metrics(self):
      """ Print all metrics """

      for _epoch in range(1, self.epoch + 1): 
      
        if self.logs['total_loss']:
            print(f"Epoch: {_epoch}, "
                  f"Total Loss: {self.logs['total_loss'][_epoch - 1]:.4f}, "
                  f"PDE Loss: {self.logs['pde_loss'][_epoch - 1]:.4f}, "
                  f"IC Loss: {self.logs['ic_loss'][_epoch - 1]:.4f}, "
                  f"BC Loss: {self.logs['bc_loss'][_epoch - 1]:.4f}, "
                  f"No Slip Loss: {self.logs['no_slip_loss'][_epoch - 1]:.4f}")
        else:
            print("No metrics to display.")

  def train_pinn(self, epochs, 
            optimizer, wing_df,
            Nf, N0, Nb, Nw,
            x_max, y_max, z_max, t_max,
            c1, c2, c3, c4,
            in_velocity,
            mu, rho,
            device,
            checkpoint_epochs,
            models_out_dir,
            filename,
            log_out_dir):
    
    while self.epoch <= epochs:

      self.epoch += 1

      optimizer.step(lambda: 
                    self.closure(
                      wing_df,
                      optimizer, 
                      Nf, N0, Nb, Nw, 
                      x_max, y_max, z_max, t_max, 
                      c1, c2, c3, c4, 
                      in_velocity, 
                      mu, 
                      rho, 
                      device))

      self.log_metrics(self.current_total_loss, self.current_pde_loss, self.current_ic_loss, self.current_bc_loss, self.current_no_slip_loss)
      self.print_current_metrics() 
      
      if np.isnan(self.current_total_loss):
        print("=> NaN loss...")
        exit(1)
        # if self.epoch % checkpoint_epochs == 0:
      #     pinn, optimizer, checkpoint_epoch = utils.load_checkpoint(pinn, optimizer, models_out_dir + filename + "_" + str(epoch - checkpoint_epochs) + ".pt")
      #   else:
      #     pinn, optimizer, checkpoint_epoch = utils.load_checkpoint(pinn, optimizer, models_out_dir + filename + "_" + str(epoch - (epoch % checkpoint_epochs)) + ".pt")
      #   epoch = checkpoint_epoch + 1
      #   continue

      if self.epoch % checkpoint_epochs == 0:
        utils.save_checkpoint(self, optimizer, models_out_dir + filename + "_" + str(self.epoch) + ".pt")