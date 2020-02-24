import math
import numpy as np


class ParticleFunction:
    def __init__(self, f):
        self.position = np.array([])
        self.velocity = 0
        self.best_position = np.array([])  # position of the best value encountered by the particle
        self.best_value = math.inf  # best value of the objective function encountered by the particle
        self.w = 0  # dynamic inertia weight to use in PSO modified formula
        self.function = f  # objective function

    def get_value(self):
        return self.function(self.position)
