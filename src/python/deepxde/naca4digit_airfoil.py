import numpy as np
import matplotlib.pyplot as plt

class Naca4DigitAirfoil:

  def __init__(self, c: float, M: float, P: float, T: float, a: float, offset_x: float, offset_y: float):
    """
    - c: Chord length.
    - M: Maximum camber as a fraction of the chord (*100).
    - P: Position of maximum camber from the leading edge as a fraction of the chord (*10).
    - T: Maximum thickness as a fraction of the chord (*100).
    - alpha_deg: Angle of attack in degrees.
    - offset_x: x-offset.
    - offset_y: y-offset.
    """
    self.c = c
    self.m = M / 100
    self.p = P / 10
    self.t = T / 100
    self.a = a
    self.offset_x = offset_x
    self.offset_y = offset_y


  def get_boundary_points(self, n: int):
    """
    Compute the boundary coordinates of a NACA 4-digits airfoil

    - n: the total points sampled will be 2*n
    """
    m = self.m
    p = self.p
    t = self.t
    c = self.c
    a = self.a
    offset_x = self.offset_x
    offset_y = self.offset_y

    if (m == 0):
        p = 1
    
    # Chord discretization (cosine discretization)
    xv = np.linspace(0.0, c, n+1)
    xv = c / 2.0 * (1.0 - np.cos(np.pi * xv / c))
    
    # Thickness distribution
    ytfcn = lambda x: 5 * t * c * (0.2969 * (x / c)**0.5 - 
                                   0.1260 * (x / c) - 
                                   0.3516 * (x / c)**2 + 
                                   0.2843 * (x / c)**3 - 
                                   0.1015 * (x / c)**4)
    yt = ytfcn(xv)
    
    # Camber line
    yc = np.zeros(np.size(xv))
    
    for i in range(n+1):
      if xv[i] <= p * c:
        yc[i] = c * (m / p**2 * (xv[i] / c) * (2 * p - (xv[i] / c)))
      else:
        yc[i] = c * (m / (1 - p)**2 * (1 + (2 * p - (xv[i] / c)) * (xv[i] / c) - 2 * p))
    
    # Camber line slope
    dyc = np.zeros(np.size(xv))
    
    for i in range(n+1):
      if xv[i] <= p * c:
        dyc[i] = m / p**2 * 2 * (p - xv[i] / c)
      else:
        dyc[i] = m / (1 - p)**2 * 2 * (p - xv[i] / c)
            
    # Boundary coordinates and sorting        
    th = np.arctan2(dyc, 1)
    xU = xv - yt * np.sin(th)
    yU = yc + yt * np.cos(th)
    xL = xv + yt * np.sin(th)
    yL = yc - yt * np.cos(th)
    
    x = np.zeros(2 * n + 1)
    y = np.zeros(2 * n + 1)
    
    for i in range(n):
      x[i] = xL[n - i]
      y[i] = yL[n - i]
        
    x[n : 2 * n + 1] = xU
    y[n : 2 * n + 1] = yU
    
    # Rotation
    ## Convert AoA to radians
    a_rad = np.radians(a)

    ## Rotation matrix
    R = np.array([[np.cos(a_rad), np.sin(a_rad)], [-np.sin(a_rad), np.cos(a_rad)]])

    ## Apply rotation
    for i in range(len(x)):
      [x[i], y[i]] = R @ np.array([x[i], y[i]])

    return np.vstack((x + offset_x, y + offset_y)).T
  
  
  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()

    points = self.get_boundary_points(1000)

    ax.fill(points[:, 0], points[:, 1], 'skyblue', label='Airfoil')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)

    if ax is None:
      plt.show()