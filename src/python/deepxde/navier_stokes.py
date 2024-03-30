import deepxde as dde
import numpy as np

class NavierStokesPDEs:

  def __init__(self, rho: float, mu: float,
               xmin: float, xmax: float, ymin: float, ymax: float,
               airfoil_geom: dde.geometry.Polygon, geom: dde.geometry.Rectangle,
               u_inlet: float = None, v_farfield: float = None,
               x_data: np.ndarray = None, y_data: np.ndarray = None, u_data: np.ndarray = None, v_data: np.ndarray = None, p_data: np.ndarray = None):
    """
    - rho: Density.
    - mu: Dynamic viscosity.
    - xmin, xmax, ymin, ymax: Domain limits.
    - airfoil_geom: Airfoil geometry.
    - u_inlet, v_farfield: Farfield boundary conditions.
    - x_data, y_data, u_data, v_data, p_data: Data points.
    """
    self.rho, self.mu = rho, mu
    self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
    self.u_inlet, self.v_farfield = u_inlet, v_farfield
    self.x_data, self.y_data, self.u_data, self.v_data, self.p_data = x_data, y_data, u_data, v_data, p_data
    self.airfoil_geom, self.geom = airfoil_geom, geom


  # def __init__(self, rho: float, mu: float,
  #              xmin: float, xmax: float, ymin: float, ymax: float,
  #              airfoil_geom: dde.geometry.Polygon, geom: dde.geometry.Rectangle,
  #              x_data: np.ndarray, y_data: np.ndarray, u_data: np.ndarray, v_data: np.ndarray, p_data: np.ndarray):
  #   """
  #   - rho: Density.
  #   - mu: Dynamic viscosity.
  #   - xmin, xmax, ymin, ymax: Domain limits.
  #   - airfoil_geom: Airfoil geometry.
  #   - x_data, y_data, u_data, v_data, p_data: Data points.
  #   """
  #   self.rho, self.mu = rho, mu
  #   self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
  #   self.x_data, self.y_data, self.u_data, self.v_data, self.p_data = x_data, y_data, u_data, v_data, p_data
  #   self.airfoil_geom, self.geom = airfoil_geom, geom


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


  def getU(self, x, y):
    return dde.grad.jacobian(y, x, i = 0, j = 1) 

  def getV(self, x, y):
    return - dde.grad.jacobian(y, x, i = 0, j = 0)  

  def getP(sefl, x, y):
    return y[:, 1:2]


  # Boundaries definition
  def __boundary_farfield_inlet(self, x, on_boundary):
    return on_boundary and np.isclose(x[0], self.xmin)

  def __boundary_farfield_top_bottom(self, x, on_boundary):
    return on_boundary and (np.isclose(x[1], self.ymax) or np.isclose(x[1], self.ymin))

  def __boundary_farfield_outlet(self, x, on_boundary):
    return on_boundary and np.isclose(x[0], self.xmax)

  def __boundary_airfoil(self, x, on_boundary):
    return on_boundary and self.airfoil_geom.on_boundary(np.array([x]))[0]


  # Boundary values definition
  def __fun_u_inlet(self, x, y, _):
    return self.getU(x, y) - self.u_inlet

  def __fun_v_farfield(self, x, y, _):
    return self.getV(x, y) - self.v_farfield

  def __fun_no_slip_u(self, x, y, _):
    return self.getU(x, y)

  def __fun_no_slip_v(self, x, y, _):
    return self.getV(x, y)

  def __fun_u_(self, x, y, _):
    return self.getU(x, y)
  
  def __fun_v_(self, x, y, _):
    return self.getV(x, y)
  
  def __fun_p_(self, x, y, _):
    return self.getP(x, y)

  # Boundary conditions assembly
  def get_bcs(self):
    bc_inlet_u = dde.OperatorBC(self.geom, self.__fun_u_inlet, self.__boundary_farfield_inlet)
    bc_inlet_v = dde.OperatorBC(self.geom, self.__fun_v_farfield, self.__boundary_farfield_inlet)

    bc_top_bottom_v = dde.OperatorBC(self.geom, self.__fun_v_farfield, self.__boundary_farfield_top_bottom)

    bc_airfoil_u = dde.OperatorBC(self.geom, self.__fun_no_slip_u, self.__boundary_airfoil)
    bc_airfoil_v = dde.OperatorBC(self.geom, self.__fun_no_slip_v, self.__boundary_airfoil)

    return [bc_inlet_u, bc_inlet_v, bc_top_bottom_v, bc_airfoil_u, bc_airfoil_v]


  # Boundary conditions with data assembly
  def get_bcs_with_data(self):

    bc_airfoil_u = dde.OperatorBC(self.geom, self.__fun_no_slip_u, self.__boundary_airfoil)
    bc_airfoil_v = dde.OperatorBC(self.geom, self.__fun_no_slip_v, self.__boundary_airfoil)

    bc_u_data = dde.PointSetOperatorBC(np.array([self.x_data, self.y_data]).T, np.array(self.u_data).reshape(-1, 1), self.__fun_u_)
    bc_v_data = dde.PointSetOperatorBC(np.array([self.x_data, self.y_data]).T, np.array(self.v_data).reshape(-1, 1), self.__fun_v_)
    # bc_p_data = dde.PointSetOperatorBC(np.array([self.x_data, self.y_data]).T, np.array(self.p_data).reshape(-1, 1), self.__fun_p_)

    # return [bc_u_data, bc_v_data, bc_p_data, bc_airfoil_u, bc_airfoil_v]
    return [bc_u_data, bc_v_data, bc_airfoil_u, bc_airfoil_v]