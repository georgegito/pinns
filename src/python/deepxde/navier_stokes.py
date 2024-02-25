import deepxde as dde


class NavierStokesPDEs:

  def __init__(self, rho: float, mu: float):
    """
    - rho: Density.
    - mu: Dynamic viscosity.
    """
    self.rho = rho
    self.mu = mu


  def get_pdes(self, x, y):
      """
      System of PDEs to be minimized: incompressible Navier-Stokes equation in the
      continuum-mechanics based formulations.
      """
      rho, mu = self.rho, self.mu

      psi, p, sigma11, sigma22, sigma12 = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]
      
      u =   dde.grad.jacobian(y, x, i = 0, j = 1)
      v = - dde.grad.jacobian(y, x, i = 0, j = 0)
      
      u_x = dde.grad.jacobian(u, x, i = 0, j = 0)
      u_y = dde.grad.jacobian(u, x, i = 0, j = 1)
      
      v_x = dde.grad.jacobian(v, x, i = 0, j = 0)
      v_y = dde.grad.jacobian(v, x, i = 0, j = 1)
      
      sigma11_x = dde.grad.jacobian(y, x, i = 2, j = 0)
      sigma12_x = dde.grad.jacobian(y, x, i = 4, j = 0)
      sigma12_y = dde.grad.jacobian(y, x, i = 4, j = 1)
      sigma22_y = dde.grad.jacobian(y, x, i = 3, j = 1)
      
      continuumx = rho * (u * u_x + v * u_y) - sigma11_x - sigma12_y
      continuumy = rho * (u * v_x + v * v_y) - sigma12_x - sigma22_y
      
      constitutive1 = - p + 2 * mu * u_x - sigma11
      constitutive2 = - p + 2 * mu * v_y - sigma22
      constitutive3 = mu * (u_y + v_x) - sigma12
      constitutive4 = p + (sigma11 + sigma22) / 2
      
      return continuumx, continuumy, constitutive1, constitutive2, constitutive3, constitutive4