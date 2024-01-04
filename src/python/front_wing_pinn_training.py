#!/usr/bin/env python
# coding: utf-8

# # Physics Informed Neural Networks <br> F1 Car Front Wing Aerodymanics

# ## PINN

# In[ ]:


import pandas as pd
import torch
from pinn import PINN
import os


# In[ ]:


data_dir = "/Users/ggito/repos/pinns/data/front_wing"
model_dir = os.path.join(data_dir, "models")

points_filename = "points_final.csv"
measurements_filename = "measurements.csv"


# In[ ]:


if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  print("GPU device not found.")

print(device)


# In[ ]:


wing_df = pd.read_csv(os.path.join(data_dir, points_filename))
measurements_df = pd.read_csv(os.path.join(data_dir, measurements_filename))


# In[ ]:


# Density (rho): 1.2041kg/m^3
# Dynamic viscosity (mu): 1.81e-5 kg/m.s
rho = 1
mu = 1

# m/s
in_velocity = 10

# Domain limits
x_max = 1
y_max = 1
z_max = 1


# In[ ]:


input_dim = 3
output_dim = 4
hidden_units = [1000, 1000, 1000]

model_name = "v30"

pinn = PINN(input_dim, output_dim, hidden_units, model_name).to(device)


# In[ ]:


optimizer = torch.optim.LBFGS(pinn.parameters(), lr=1, line_search_fn="strong_wolfe")


# In[ ]:


epochs = 3000
checkpoint_epochs = 50

Nf = 100   # num of collocation points -> pde evaluation
Nb = 100   # num of points to evaluate boundary conditions
Nw = 100   # num of points of the surface of the front wing to evaluate boundary conditions
Nu = 2     # num of points of real data


# In[ ]:


pinn.train_pinn(
    epochs=epochs, 
    optimizer=optimizer, 
    s_df=wing_df, 
    u_df=measurements_df, 
    Nf=Nf, 
    Nb=Nb, 
    Ns=Nw, 
    Nu=Nu, 
    x_max=x_max, 
    y_max=y_max, 
    z_max=z_max, 
    in_velocity=in_velocity, 
    mu=mu, 
    rho=rho, 
    device=device, 
    checkpoint_epochs=checkpoint_epochs, 
    model_dir=model_dir,
    c2=100, c3=100, c4=100, c5=100, c6=100, c8=100, c9=100,
)