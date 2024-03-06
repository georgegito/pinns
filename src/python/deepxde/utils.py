import random

class NameGenerator:
  def __init__(self):
    self.tech_words = ['Aero', 'Atmos', 'JetStream', 'Vortex', 'Wind', 'Gust', 'Breeze', 'AirFlow', 'Sky', 'Stratos']
    self.adjective_words = ['Precise', 'Accurate', 'Analytic', 'Dynamic', 'Elegant', 'Robust', 'Optimized', 'Advanced', 'Intelligent', 'Rapid']
    self.futuristic_words = ['Voyager', 'Frontier', 'Horizon', 'Innovator', 'Pioneer', 'Navigator', 'Quantica', 'Cosmos', 'Nebula', 'Galaxy']

  def generate_name(self):
    code = f"{random.randint(0, 999):03d}"
    return ''.join([
      random.choice(self.tech_words), 
      random.choice(self.adjective_words), 
      random.choice(self.futuristic_words),
      code])

