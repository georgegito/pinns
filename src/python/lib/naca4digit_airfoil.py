import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import utils

class Naca4DigitAirfoil:

  def __init__(self, chord: float, m: float, p: float, t: float, alpha_deg = 0, num_points: int=1000):
    """
    - chord: Chord length.
    - m: Maximum camber as a fraction of the chord.
    - p: Position of maximum camber from the leading edge as a fraction of the chord.
    - t: Maximum thickness as a fraction of the chord.
    """
    self.chord = chord
    self.m = m
    self.p = p
    self.t = t

    self.xu, self.yu, self.xl, self.yl = self.__generate_surface_points(num_points ,alpha_deg)

    self.x_min = self.xu.min()
    self.x_max = self.xu.max()
    self.y_min = self.yl.min()
    self.y_max = self.yu.max()


  def plot(self, ax=None):
    """
    Plot the upper and lower surfaces of an airfoil and fill the area between them.

    Parameters:
    - ax: Optional matplotlib axes object to plot on. If None, creates a new figure.

    """
    if ax is None:
      fig, ax = plt.subplots()  # Create new figure and axes if not provided

    ax.plot(self.xu, self.yu, 'black', linewidth=0.5)
    ax.plot(self.xl, self.yl, 'black', linewidth=0.5)
    ax.fill_between(self.xu, self.yu, self.yl, color='skyblue', label='Airfoil')
    ax.axis('equal')
    ax.legend()
    ax.set_xlabel('Chord Length')
    ax.set_ylabel('Vertical Distance')
    ax.set_title('NACA 4-Digit Airfoil')
    ax.grid(True)

    # If you created a new figure, show it; otherwise, let the caller handle showing or further modification.
    if ax is None:
      plt.show()


  def __generate_surface_points(self, num_points: int, alpha_deg: float) -> tuple:
    """
    Generate the surface coordinates of a NACA 4-digit airfoil at a specified angle of attack.

    Parameters:
    - num_points: Number of points to sample on each surface (upper and lower).
    - alpha_deg: Angle of attack in degrees.

    Returns:
    - xu_r, yu_r, xl_r, yl_r: Rotated coordinates of the airfoil.
    """
    x = np.linspace(0, self.chord, num_points)
    yt = 5*self.t*self.chord*(0.2969*np.sqrt(x/self.chord) - 0.1260*(x/self.chord) - 0.3516*(x/self.chord)**2 + 0.2843*(x/self.chord)**3 - 0.1015*(x/self.chord)**4)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i in range(len(x)):
      if x[i] < self.p*self.chord:
        yc[i] = (self.m/self.p**2)*(2*self.p*(x[i]/self.chord) - (x[i]/self.chord)**2)
        dyc_dx[i] = (2*self.m/self.p**2)*(self.p - x[i]/self.chord)
      else:
        yc[i] = (self.m/(1-self.p)**2)*((1-2*self.p) + 2*self.p*(x[i]/self.chord) - (x[i]/self.chord)**2)
        dyc_dx[i] = (2*self.m/(1-self.p)**2)*(self.p - x[i]/self.chord)
  
    theta = np.arctan(dyc_dx)
    
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)

    # Convert AoA to radians
    alpha_rad = np.radians(alpha_deg)

    # Rotation matrix
    R = np.array([[np.cos(alpha_rad), np.sin(alpha_rad)], [-np.sin(alpha_rad), np.cos(alpha_rad)]])

    # Initialize rotated coordinates
    xu_r, yu_r, xl_r, yl_r = np.zeros_like(xu), np.zeros_like(yu), np.zeros_like(xl), np.zeros_like(yl)

    # Apply rotation
    for i in range(len(xu)):
      [xu_r[i], yu_r[i]] = R @ np.array([xu[i], yu[i]])
      [xl_r[i], yl_r[i]] = R @ np.array([xl[i], yl[i]])

    return xu_r, yu_r, xl_r, yl_r
  
  
  def generate_interior_points(self, num_interior_points=1000) -> tuple:
    """
    Generate points inside the airfoil by interpolating between the upper and lower surfaces.

    Parameters:
    - num_interior_points: Number of interior points to generate.

    Returns:
    - x_points, y_points: Coordinates of the points inside the airfoil.
    """
    x_in = []
    y_in = []
    
    for _ in range(num_interior_points):
      # Choose a random x-coordinate
      x = np.random.uniform(low=min(self.xu), high=max(self.xu))
      
      # Interpolate to find the corresponding y-values on the upper and lower surfaces
      y_upper = np.interp(x, self.xu, self.yu)
      y_lower = np.interp(x, self.xl, self.yl)
      
      # Generate a random y-coordinate between the upper and lower y-values
      y = np.random.uniform(low=y_lower, high=y_upper)
      
      x_in.append(x)
      y_in.append(y)
    
    return np.array(x_in), np.array(y_in)


  def classify_points(self, points: np.ndarray) -> tuple:
    """
    Classifies pre-sampled points as either interior or exterior relative to an airfoil.

    Parameters:
    - points: Array of points to classify, shaped as (n, 2), where n is the number of points,
              and each point is [x, y].

    Returns:
    - interior_points: Points that are inside the airfoil.
    - exterior_points: Points that are outside the airfoil.
    """

    interior_points = []
    exterior_points = []

    for x, y in points:
      y_upper = np.interp(x, self.xu, self.yu, left=np.nan, right=np.nan)
      y_lower = np.interp(x, self.xl, self.yl, left=np.nan, right=np.nan)
      
      if np.isnan(y_upper) or np.isnan(y_lower):
        exterior_points.append([x, y])
      elif y_lower <= y <= y_upper:
        interior_points.append([x, y])
      else:
        exterior_points.append([x, y])
    
    return np.array(interior_points), np.array(exterior_points)
  
  
  def sample_surface_points(self, num_points: int) -> tuple:
    """
    Sample points on the upper and lower surfaces of the airfoil.

    Parameters:
    - num_points: Number of points to sample on each surface (upper and lower).

    Returns:
    - x, y: Coordinates of the airfoil.
    """
    # Define interpolation functions for upper and lower surfaces
    interp_upper = interp1d(self.xu, self.yu, kind='cubic')
    interp_lower = interp1d(self.xl, self.yl, kind='cubic')

    # Sample new x positions
    new_x_positions = np.linspace(self.x_min+0.01, self.x_max-0.01, num_points) # More finely spaced x points

    # Use the interpolation functions to get new y values
    new_yu = interp_upper(new_x_positions)
    new_yl = interp_lower(new_x_positions)

    x = np.concatenate((new_x_positions, new_x_positions))
    y = np.concatenate((new_yu, new_yl))

    return x, y
  
  def sample_points_in_domain_around(self, num_points: int, distance: float) -> tuple:
    """
    Sample points in the domain around the airfoil.

    Parameters:
    - num_points: Number of points to sample.
    - distance: Distance around the airfoil to sample.

    Returns:
    - x, y: Coordinates of the points in the domain around the airfoil.
    """
    return utils.qmc_sample_points_in_domain_2d(self.x_min - distance, self.x_max + distance, self.y_min - distance, self.y_max + distance, num_points) 