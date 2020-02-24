import test_functions
import pso
from ParticleFunctionFactory import ParticleFunctionFactory

particleFactory = ParticleFunctionFactory(test_functions.himmelblau)
min_point, min_value = pso.get_minimum(particleFactory, 2, [(-10, 10), (-10, 10)])

print("Minimum point: " + str(min_point))
print("Minimum value: " + str(min_value))

