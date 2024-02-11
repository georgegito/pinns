import numpy as np
import matplotlib.pyplot as plt

class Naca4DigitAirfoil:

  def __init__(self, chord: float, m: float, p: float, t: float, num_points: int=1000):
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

    self.xu, self.yu, self.xl, self.yl = self.__generate_surface_points(num_points)


  def plot(self, ax=None):
    """
    Plot the upper and lower surfaces of an airfoil and fill the area between them.

    Parameters:
    - ax: Optional matplotlib axes object to plot on. If None, creates a new figure.

    """
    if ax is None:
      fig, ax = plt.subplots()  # Create new figure and axes if not provided

    ax.plot(self.xu, self.yu, 'r-', label='Upper Surface')
    ax.plot(self.xl, self.yl, 'b-', label='Lower Surface')
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


  def __generate_surface_points(self, num_points: int) -> tuple:
    """
    Generate the surface coordinates of a NACA 4-digit airfoil.

    Parameters:
    - num_points: Number of points to sample on each surface (upper and lower).

    Returns:
    - xu, yu, xl, yl: Coordinates of the airfoil.
    """
    x = np.linspace(0, self.chord, num_points)
    
    # Thickness distribution
    yt = 5*self.t*self.chord*(0.2969*np.sqrt(x/self.chord) - 0.1260*(x/self.chord) - 0.3516*(x/self.chord)**2 + 0.2843*(x/self.chord)**3 - 0.1015*(x/self.chord)**4)
    
    # Camber line
    yc = np.zeros_like(x)
    for i in range(len(x)):
      if x[i] < self.p*self.chord:
        yc[i] = (self.m/self.p**2)*(2*self.p*(x[i]/self.chord) - (x[i]/self.chord)**2)
      else:
        yc[i] = (self.m/(1-self.p)**2)*((1-2*self.p) + 2*self.p*(x[i]/self.chord) - (x[i]/self.chord)**2)

    # Camber line slope
    dyc_dx = np.zeros_like(x)
    for i in range(len(x)):
      if x[i] < self.p*self.chord:
        dyc_dx[i] = (2*self.m/self.p**2)*(self.p - x[i]/self.chord)
      else:
        dyc_dx[i] = (2*self.m/(1-self.p)**2)*(self.p - x[i]/self.chord)
  
    theta = np.arctan(dyc_dx)
    
    # Upper and lower surfaces
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    
    return xu, yu, xl, yl
  
  
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