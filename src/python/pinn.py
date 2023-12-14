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

  def grad(self, x, y, create_graph=True):
    return torch.autograd.grad(x, y, grad_outputs=torch.ones_like(x), create_graph=create_graph, retain_graph=True, only_inputs=True)[0]

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
        input_f, 
        input_0, 
        input_b,
        input_w,
        normals_w,
        in_velocity,
        mu, rho, c1, c2, c3, c4, c5,
        log_filepath):

    output_f = self(input_f)
    u = output_f[:, 0]
    v = output_f[:, 1]
    w = output_f[:, 2]
    p = output_f[:, 3]

    x_f = input_f[:, 0]
    y_f = input_f[:, 1]
    z_f = input_f[:, 2]
    t_f = input_f[:, 3]
     
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

    # b = rho * ( 1/dt * (u_x + v_y) - u_x**2 - 2*u_y*v_x - v_y**2)

    f1 = u_t + u*u_x + v*u_y + w*u_z + (1/rho) * p_x - mu * (u_xx + u_yy + u_zz)
    f2 = v_t + u*v_x + v*v_y + w*v_z + (1/rho) * p_y - mu * (v_xx + v_yy + v_zz)
    f3 = w_t + u*w_x + v*w_y + w*w_z + (1/rho) * p_z - mu * (w_xx + w_yy + w_zz)
    f3 = u_x + v_y + w_z

    pde_loss =  torch.mean(torch.square(f1)) + \
                torch.mean(torch.square(f2)) + \
                torch.mean(torch.square(f3)) / 3

    # TODO: add poisson equation
    # f4 = p_xx + p_yy + p_zz - b

    # Initial condition loss
    output_0 = self(input_0)
    u0_pred = output_0[:, 0]
    v0_pred = output_0[:, 1]
    w0_pred = output_0[:, 2]
    p0_pred = output_0[:, 3]

    # for t = 0 -> u, v, w = 0, p = 1

    u0_true = torch.zeros_like(u0_pred)
    v0_true = torch.zeros_like(v0_pred)
    w0_true = torch.zeros_like(w0_pred)
    p0_true = torch.ones_like(p0_pred)

    ic_loss_u = torch.mean(torch.square(u0_pred - u0_true))
    ic_loss_v = torch.mean(torch.square(v0_pred - v0_true))
    ic_loss_w = torch.mean(torch.square(w0_pred - w0_true))
    ic_loss_p = torch.mean(torch.square(p0_pred - p0_true))

    ic_loss = (ic_loss_u + ic_loss_v + ic_loss_w + ic_loss_p) / 3

    # Boundary conditions loss

    # xyzt_combinations = torch.cartesian_prod(x_b.flatten(), y_b.flatten(), z_b.flatten(), t_b.flatten()) # TODO
    output_b = self(input_b)
    u_b_pred = output_b[:, 0]
    v_b_pred = output_b[:, 1]
    w_b_pred = output_b[:, 2]

    # u = 0, v = -1 * in_velocity and w = 0 for y = 1

    u_b_true = torch.zeros_like(u_b_pred) # TODO
    v_b_true = torch.full_like(v_b_pred, -1 * in_velocity)
    w_b_true = torch.zeros_like(w_b_pred)
    
    bc_loss_u = torch.mean(torch.square(u_b_pred - u_b_true))
    bc_loss_v = torch.mean(torch.square(v_b_pred - v_b_true))
    bc_loss_w = torch.mean(torch.square(w_b_pred - w_b_true))

    bc_loss = (bc_loss_u + bc_loss_v + bc_loss_w) / 3

    # Wing surface boundary conditions loss
    
    # xyzt_combinations = torch.cartesian_prod(x_w.flatten(), y_w.flatten(), z_w.flatten(), t_w.flatten()) # TODO

    # xyz_stacked = torch.stack((x_w, y_w, z_w), dim=-1)

    # print(xyz_stacked.shape)

    # xyzt_combinations = torch.cartesian_prod(xyz_stacked, t_w)

    # print(xyzt_combinations.shape)

    output_wing = self(input_w)
    u_w_pred = output_wing[:, 0]
    v_w_pred = output_wing[:, 1]
    w_w_pred = output_wing[:, 2]

    ## no-slip condition
    no_slip_loss_u = torch.mean(torch.square(u_w_pred))
    no_slip_loss_v = torch.mean(torch.square(v_w_pred))
    no_slip_loss_w = torch.mean(torch.square(w_w_pred))

    no_slip_loss = (no_slip_loss_u + no_slip_loss_v + no_slip_loss_w) / 3

    # TODO
    # ## impermeability condition
    # print(u_w_pred.shape, n_x.shape)

    # imp_residual = u_w_pred * n_x + v_w_pred * n_y + w_w_pred * n_z

    # imp_loss = torch.mean(torch.square(imp_residual))
    imp_loss = 0

    # total loss
    total_loss =  c1 * pde_loss + \
                  c2 * ic_loss + \
                  c3 * bc_loss + \
                  c4 * no_slip_loss + \
                  c5 * imp_loss

    self.save_log(log_filepath, total_loss, pde_loss, ic_loss, bc_loss, no_slip_loss, imp_loss)

    return total_loss, pde_loss, ic_loss, bc_loss, no_slip_loss, imp_loss