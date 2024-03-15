import deepxde as dde
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from naca4digit_airfoil import Naca4DigitAirfoil
from navier_stokes_with_data import NavierStokesPDEs
import utils
import pandas as pd
from scipy.interpolate import griddata

cfd_df = pd.read_csv('NACA2412.txt', delimiter=',')

x_min = -0.5
x_max = 2
y_min = -1
y_max = 1

# sf_xy = max(abs(cfd_df.x).max(), abs(cfd_df.y).max())
sf_xy = max(x_max - x_min, y_max - y_min)
sf_uv = max(abs(cfd_df.u).max(), abs(cfd_df.v).max())
sf_p = abs(cfd_df).p.max()

x_min_scaled = x_min / sf_xy
x_max_scaled = x_max / sf_xy
y_min_scaled = y_min / sf_xy
y_max_scaled = y_max / sf_xy

dde.config.set_random_seed(48)
dde.config.set_default_float('float64')

rho  = 1.225
mu   = 1.81e-5
# u_in  = 15
# u_in  = 1

airfoil = Naca4DigitAirfoil(c=1/sf_xy, M=2, P=4, T=12, a=0, offset_x=0, offset_y=0)

# Geometry defintion
farfield = dde.geometry.Rectangle([x_min_scaled, y_min_scaled], [x_max_scaled, y_max_scaled])
airfoil_geom  = dde.geometry.Polygon(airfoil.get_boundary_points(250))
geom     = dde.geometry.CSGDifference(farfield, airfoil_geom)

# inner_rec  = dde.geometry.Rectangle([-0.1, -0.1], [0.1, 0.1])
inner_rec  = dde.geometry.Rectangle([-0.5/sf_xy, -0.5/sf_xy], [0.5/sf_xy, 0.5/sf_xy])
# inner_rec  = dde.geometry.Rectangle([-0.05, -0.05], [0.05, 0.05])

inner_dom  = dde.geometry.CSGDifference(inner_rec, airfoil_geom)

outer_dom  = dde.geometry.CSGDifference(farfield, inner_rec)
outer_dom  = dde.geometry.CSGDifference(outer_dom, airfoil_geom)

Nf1 = 2**15 # = 32768
Nf2 = 2**16 # = 65536
Nb  = 2**11 # = 2048
Ns  = 250

random = "Sobol"
inner_points = inner_dom.random_points(Nf1, random=random)
outer_points = outer_dom.random_points(Nf2, random=random)

farfield_points = farfield.random_boundary_points(Nb, random=random)
airfoil_points  = airfoil.get_boundary_points(Ns)

points = np.append(inner_points, outer_points, axis = 0)
# points = np.append(points, farfield_points, axis = 0)
points = np.append(points, airfoil_points, axis = 0)

x_data = farfield_points[:, 0]
y_data = farfield_points[:, 1]

grid_points = cfd_df[['x', 'y']].values / sf_xy
u_values = cfd_df['u'].values / sf_uv
v_values = cfd_df['v'].values / sf_uv
p_values = cfd_df['p'].values / sf_p

u_interp = griddata(grid_points, u_values, (x_data, y_data), method='linear')
v_interp = griddata(grid_points, v_values, (x_data, y_data), method='linear')
p_interp = griddata(grid_points, p_values, (x_data, y_data), method='linear')

uvp_data = np.vstack((u_interp, v_interp, p_interp)).T

u_data = np.array([uvp_data[i][0] for i in range(len(uvp_data))])
v_data = np.array([uvp_data[i][1] for i in range(len(uvp_data))])
p_data = np.array([uvp_data[i][2] for i in range(len(uvp_data))])


navier_stokes_pdes = NavierStokesPDEs(rho=rho, mu=mu,
                                      xmin=x_min_scaled, xmax=x_max_scaled, ymin=y_min_scaled, ymax=y_max_scaled,
                                      airfoil_geom=airfoil_geom, geom=geom,
                                      x_data=x_data, y_data=y_data, u_data=u_data, v_data=v_data, p_data=p_data)

pdes_fun = navier_stokes_pdes.get_pdes
bcs = navier_stokes_pdes.get_bcs()


# Problem setup
data = dde.data.PDE(geom, pdes_fun, bcs, num_domain = 0, num_boundary = 0, num_test = 5000, anchors = points)

dde.config.set_default_float('float64')

# Neural network definition
layer_size  = [2] + [32] * 10 + [5]
# activation  = 'silu' 
activation  = 'tanh' 
initializer = 'Glorot uniform'

net = dde.nn.FNN(layer_size, activation, initializer)

# Model definition
model = dde.Model(data, net)

name_generator = utils.NameGenerator()
model_name = name_generator.generate_name()


# model.compile(optimizer = 'adam', lr = 5e-4, loss_weights = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]) # Giving more weight to bcs
model.compile(optimizer = 'adam', lr = 5e-4)


try:
  losshistory, train_state = model.train(epochs = 200, display_every = 100, model_save_path = './' + model_name + '/')
  dde.saveplot(losshistory, train_state, issave = True, isplot = True)

except KeyboardInterrupt:
  print(f"Training stopped by user.")
  print("=======================================================")


# Plotting tool: thanks to @q769855234 code snippet
dx = 0.1
dy = 0.1
x = np.arange(x_min_scaled, x_max_scaled + dy, dx)
y = np.arange(y_min_scaled, y_max_scaled + dy, dy)

X = np.zeros((len(x)*len(y), 2))
xs = np.vstack((x,)*len(y)).reshape(-1)
ys = np.vstack((y,)*len(x)).T.reshape(-1)
X[:, 0] = xs
X[:, 1] = ys

# Model predictions generation
u = model.predict(X, operator = navier_stokes_pdes.getU)
v = model.predict(X, operator = navier_stokes_pdes.getV)
p = model.predict(X, operator = navier_stokes_pdes.getP)

for i in range(len(X)):
   if airfoil_geom.inside(np.array([X[i]]))[0]:
       u[i] = 0.0
       v[i] = 0.0

u = u.reshape(len(y), len(x))
v = v.reshape(len(y), len(x))
p = p.reshape(len(y), len(x))

airfoil_plot = airfoil.get_boundary_points(150)

fig1, ax1 = plt.subplots(figsize = (16, 9))
#ax1.streamplot(x, y, u, v, density = 1.5)
clev = np.arange(p.min(), p.max(), 0.001)
cnt1 = ax1.contourf(x, y, p, clev, cmap = plt.cm.jet, extend='both')
plt.axis('equal')
plt.fill(airfoil_plot[:, 0], airfoil_plot[:, 1])
fig1.colorbar(cnt1)
plt.savefig(model_name + '/p.png')

fig2, ax2 = plt.subplots(figsize = (16, 9))
ax2.streamplot(x, y, u, v, density = 1.5)
clev = np.arange(u.min(), u.max(), 0.001)
cnt2 = ax2.contourf(x, y, u, clev, cmap = plt.cm.jet, extend='both')
plt.axis('equal')
plt.fill(airfoil_plot[:, 0], airfoil_plot[:, 1])
fig2.colorbar(cnt2)
plt.savefig(model_name + '/u.png')

fig3, ax3 = plt.subplots(figsize = (16, 9))
ax3.streamplot(x, y, u, v, density = 1.5)
clev = np.arange(v.min(), v.max(), 0.001)
cnt3 = ax3.contourf(x, y, v, clev, cmap = plt.cm.jet, extend='both')
plt.axis('equal')
plt.fill(airfoil_plot[:, 0], airfoil_plot[:, 1])
fig3.colorbar(cnt3)
plt.savefig(model_name + '/v.png')
