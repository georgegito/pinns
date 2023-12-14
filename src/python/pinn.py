import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv

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

  def forward(self, input):
    for layer in self.layers[:-1]:
      # output = torch.sigmoid(layer(input))
      output = torch.tanh(layer(input))
      # output = torch.relu(layer(input))
      input = output
    output = self.layers[-1](input)
    return output

  def grad(self, x, y):
    return torch.autograd.grad(x, y, grad_outputs=torch.ones_like(x), create_graph=True, retain_graph=True, only_inputs=True)[0]

  def save_log(self, log_filepath, total_loss, pde_loss, ic_loss, bc_loss, wing_loss):
    new_row = {
      "total_loss": total_loss.item(),
      "pde_loss": pde_loss.item(),
      "ic_loss": ic_loss.item(),
      "bc_loss": bc_loss.item(),
      "wing_loss": wing_loss.item()
    }

    with open(log_filepath, 'a', newline='') as file:
        
      writer = csv.DictWriter(file, fieldnames=new_row.keys())
      
      file.seek(0, 2)
      if file.tell() == 0:
        writer.writeheader()
      
      writer.writerow(new_row)

  def loss(self, 
        x_f, y_f, z_f, t_f, 
        x0, y0, z0, t0, 
        x_b, y_b, z_b, t_b,
        x_w, y_w, z_w, t_w,
        mu, rho, dt, c1, c2, c3, c4, log_filepath):

    xyzt_combinations = torch.cartesian_prod(x_f.flatten(), y_f.flatten(), z_f.flatten(), t_f.flatten())
    output = self(xyzt_combinations)
    u = output[:, 0]
    v = output[:, 1]
    w = output[:, 2]
    p = output[:, 3]

    u_t = self.grad(u, t_f)
    u_x = self.grad(u, x_f)
    u_y = self.grad(u, y_f)
    u_z = self.grad(u, z_f)
    u_xx = self.grad(u_x, x_f)
    u_yy = self.grad(u_y, y_f)
    u_zz = self.grad(u_z, z_f)

    v_t = self.grad(v, t_f)
    v_x = self.grad(v, x_f)
    v_y = self.grad(v, y_f)
    v_z = self.grad(v, z_f)
    v_xx = self.grad(v_x, x_f)
    v_yy = self.grad(v_y, y_f)
    v_zz = self.grad(v_z, z_f)

    w_t = self.grad(w, t_f)
    w_x = self.grad(w, x_f)
    w_y = self.grad(w, y_f)
    w_z = self.grad(w, z_f)
    w_xx = self.grad(w_x, x_f)
    w_yy = self.grad(w_y, y_f)
    w_zz = self.grad(w_z, z_f)

    p_x = self.grad(p, x_f)
    p_xx = self.grad(p_x, x_f)
    p_y = self.grad(p, y_f)
    p_yy = self.grad(p_y, y_f)
    p_z = self.grad(p, z_f)
    p_zz = self.grad(p_z, z_f)

    # b = rho * ( 1/dt * (u_x + v_y) - u_x**2 - 2*u_y*v_x - v_y**2)

    f1 = u_t + u*u_x + v*u_y + w*u_z + (1/rho) * p_x - mu * (u_xx + u_yy + u_zz)
    f2 = v_t + u*v_x + v*v_y + w*v_z + (1/rho) * p_y - mu * (v_xx + v_yy + v_zz)
    f3 = w_t + u*w_x + v*w_y + w*w_z + (1/rho) * p_z - mu * (w_xx + w_yy + w_zz)
    f3 = u_x + v_y + w_z
    # TODO: add poisson equation & impermeability condition
    # f4 = p_xx + p_yy + p_zz - b

    # Initial condition loss
    output_init = self(torch.cat([x0, y0, z0, t0], dim=1))
    u0_pred = output_init[:, 0]
    v0_pred = output_init[:, 1]
    w0_pred = output_init[:, 2]
    p0_pred = output_init[:, 3]

    # for x > 0 and t = 0 -> u, v, p = 0

    u0_true = torch.zeros_like(u0_pred)
    v0_true = torch.zeros_like(v0_pred)
    w0_true = torch.zeros_like(w0_pred)
    p0_true = torch.ones_like(p0_pred)

    ic_loss_u = torch.mean(torch.square(u0_pred - u0_true))
    ic_loss_v = torch.mean(torch.square(v0_pred - v0_true))
    ic_loss_w = torch.mean(torch.square(w0_pred - w0_true))
    ic_loss_p = torch.mean(torch.square(p0_pred - p0_true))

    # Boundary conditions loss

    xyzt_combinations = torch.cartesian_prod(x_b.flatten(), y_b.flatten(), z_b.flatten(), t_b.flatten())
    output_boundary = self(xyzt_combinations)
    u_b_pred = output_boundary[:, 0]
    v_b_pred = output_boundary[:, 1]
    w_b_pred = output_boundary[:, 2]

    # u = 0, v = -1 and w = 0 for x = 0

    u_b_true = torch.zeros_like(u_b_pred) # TODO
    v_b_true = torch.full_like(-1, v_b_pred)
    w_b_true = torch.zeros_like(w_b_pred)
    
    bc_loss_u = torch.mean(torch.square(u_b_pred - u_b_true))
    bc_loss_v = torch.mean(torch.square(v_b_pred - v_b_true))
    bc_loss_w = torch.mean(torch.square(w_b_pred - w_b_true))

    # Wing surface boundary conditions loss

    xyzt_combinations = torch.cartesian_prod(x_w.flatten(), y_w.flatten(), z_w.flatten(), t_w.flatten())
    output_wing = self(xyzt_combinations)
    u_w_pred = output_wing[:, 0]
    v_w_pred = output_wing[:, 1]
    w_w_pred = output_wing[:, 2]

    u_w_true = torch.zeros_like(u_w_pred)
    v_w_true = torch.zeros_like(v_w_pred)
    w_w_true = torch.zeros_like(w_w_pred)
    
    wing_loss_u = torch.mean(torch.square(u_w_pred - u_w_true))
    wing_loss_v = torch.mean(torch.square(v_w_pred - v_w_true))
    wing_loss_w = torch.mean(torch.square(w_w_pred - w_w_true))

    # Combine PDE residual, initial condition, and boundary condition losses
    pde_loss =  torch.mean(torch.square(f1)) + \
                torch.mean(torch.square(f2)) + \
                torch.mean(torch.square(f3)) / 3
    
    ic_loss = (ic_loss_u + ic_loss_v + ic_loss_w + ic_loss_p) / 3 
    
    bc_loss = (bc_loss_u + bc_loss_v + bc_loss_w) / 3

    wing_loss = (wing_loss_u + wing_loss_v + wing_loss_w) / 3

    total_loss =  c1 * pde_loss + \
                  c2 * ic_loss + \
                  c3 * bc_loss + \
                  c4 * wing_loss

    self.save_log(log_filepath, total_loss, pde_loss, ic_loss, bc_loss, wing_loss)

    return total_loss, pde_loss, ic_loss, bc_loss, wing_loss