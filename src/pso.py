import random
import sobol_seq
import math
import numpy as np
import logging
from numpy import linalg as LA
from enum import Enum


class InitializationType(Enum):
    QUASI_RANDOM = 0
    QUASI_RANDOM_USING_BORDER = 1


# PSO Hyperparameters
class PSOHyperparameters:

    def __init__(self, n):
        """
        :param n: Dimension of the problem to be resolved by te pso algorithm
        """
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 0.5
        self.c2 = 0.5
        self.swarm_size = 10
        self.num_generations = 10
        self.max_velocity = [5] * n
        self.initialization_type = InitializationType.QUASI_RANDOM
        # True = perform memetic variant of the algorithm
        self.use_local_search = False


# Number of function evalutations performed by the PSO
n_function_evaluations = 0


def get_minimum(particle_factory, n, bounds, pso_hyperparameters=None):
    global n_function_evaluations
    n_function_evaluations = 0

    # Number of bounds must be equal to the problem dimension (number of problem hyperparameters)
    assert len(bounds) == n

    if pso_hyperparameters is None:
        pso_hyperparameters = PSOHyperparameters(n)

    w_start = pso_hyperparameters.w_start
    w_end = pso_hyperparameters.w_end
    c1 = pso_hyperparameters.c1
    c2 = pso_hyperparameters.c2
    swarm_size = pso_hyperparameters.swarm_size
    num_generations = pso_hyperparameters.num_generations
    max_velocity = pso_hyperparameters.max_velocity
    initialization_type = pso_hyperparameters.initialization_type
    use_local_search = pso_hyperparameters.use_local_search

    particles = []
    global_best_position = None
    global_best_value = math.inf

    # Random initialize the particles and evaluate them
    print("\n\n***** Particles initialization *****")
    logging.debug("\n\n***** Particles initialization *****")

    particles_to_initialize = swarm_size

    if initialization_type == InitializationType.QUASI_RANDOM_USING_BORDER:
        # 2^n punti vengono inizializzati agli estremi della regione ammissibile (ipercubo)
        # JR Il codice va generalizzato ma ho bisogno di fare delle prove immediate

        assert n == 2

        particles_to_initialize = swarm_size - 4

        # Vertice 0 del quadrato (ipercubo con n = 2)
        particle = particle_factory.get_particle()
        particle.position = np.array([bounds[0][0], bounds[1][0]])
        particle.velocity = np.zeros(n)
        particle.position_integer = np.around(particle.position)
        particle.w = w_start
        particle_value = particle.get_value()

        particle.best_position = np.array(particle.position)
        particle.best_value = particle_value
        if particle_value < global_best_value:
            global_best_value = particle_value
            global_best_position = np.array(particle.position_integer)

        particles.append(particle)

        print("Border Particle 0 - Position " + str(particle.position))
        logging.debug("\nBorder Particle 0")
        logging.debug("Position : " + str(particle.position))
        logging.debug("Velocity : " + str(particle.velocity))
        logging.debug("Objective function value : " + str(particle_value))

        # Vertice 1
        particle = particle_factory.get_particle()
        particle.position = np.array([bounds[0][0], bounds[1][1]])
        particle.velocity = np.zeros(n)
        particle.position_integer = np.around(particle.position)
        particle.w = w_start
        particle_value = particle.get_value()

        particle.best_position = np.array(particle.position)
        particle.best_value = particle_value
        if particle_value < global_best_value:
            global_best_value = particle_value
            global_best_position = np.array(particle.position_integer)

        particles.append(particle)

        print("Border Particle 1 - Position " + str(particle.position))
        logging.debug("\nBorder Particle 1")
        logging.debug("Position : " + str(particle.position))
        logging.debug("Velocity : " + str(particle.velocity))
        logging.debug("Objective function value : " + str(particle_value))

        # Vertice 2
        particle = particle_factory.get_particle()
        particle.position = np.array([bounds[0][1], bounds[1][0]])
        particle.velocity = np.zeros(n)
        particle.position_integer = np.around(particle.position)
        particle.w = w_start
        particle_value = particle.get_value()

        particle.best_position = np.array(particle.position)
        particle.best_value = particle_value
        if particle_value < global_best_value:
            global_best_value = particle_value
            global_best_position = np.array(particle.position_integer)

        particles.append(particle)

        print("Border Particle 2 - Position " + str(particle.position))
        logging.debug("\nBorder Particle 2")
        logging.debug("Position : " + str(particle.position))
        logging.debug("Velocity : " + str(particle.velocity))
        logging.debug("Objective function value : " + str(particle_value))

        # Vertice 3
        particle = particle_factory.get_particle()
        particle.position = np.array([bounds[0][1], bounds[1][1]])
        particle.velocity = np.zeros(n)
        particle.position_integer = np.around(particle.position)
        particle.w = w_start
        particle_value = particle.get_value()

        particle.best_position = np.array(particle.position)
        particle.best_value = particle_value
        if particle_value < global_best_value:
            global_best_value = particle_value
            global_best_position = np.array(particle.position_integer)

        particles.append(particle)

        print("Border Particle 3 - Position " + str(particle.position))
        logging.debug("\nBorder Particle 3")
        logging.debug("Position : " + str(particle.position))
        logging.debug("Velocity : " + str(particle.velocity))
        logging.debug("Objective function value : " + str(particle_value))

    for i in range(particles_to_initialize):
        # Initial velocity is 0 according to several papers
        velocity = np.zeros(n)

        # Initial position is formed by quasi random numbers given by a Sobol sequence
        position = np.array(sobol_seq.i4_sobol(n, i + 1)[0])

        for j in range(n):
            # Put the position values into the bounds range
            position[j] = bounds[j][0] + position[j] * (bounds[j][1] - bounds[j][0])

        particle = particle_factory.get_particle()
        particle.position = position
        particle.position_integer = np.around(position)
        particle.velocity = velocity
        particle.w = w_start

        particle_value = particle.get_value()

        particle.best_position = np.array(position)
        particle.best_value = particle_value

        if particle_value < global_best_value:
            global_best_value = particle_value
            global_best_position = np.array(particle.position_integer)

        print("Particle " + str(i) + " - Position " + str(particle.position))
        logging.debug("\nParticle " + str(i))
        logging.debug("Position : " + str(particle.position))
        logging.debug("Velocity : " + str(particle.velocity))

        particles.append(particle)

    # PSO Optimization
    for i in range(num_generations):
        print("\n\n**** Particle generation " + str(i))
        logging.debug("\n\n***** Particle generation " + str(i) + " ******")

        # Auxiliary variables to temporary store best population indexes (the update of the best indexes is done after the particles update)
        global_best_position_i = global_best_position
        global_best_value_i = global_best_value

        for j in range(swarm_size):
            logging.debug("\nParticle " + str(j))

            # Move particle
            particle = particles[j]

            # Update the random hyperparameters
            r1 = np.random.rand()
            r2 = np.random.rand()

            # Particle swarm optimization update formulae
            velocity_updated = particle.w * particle.velocity + c1 * r1 * \
                               (particle.best_position - particle.position) + c2 * r2 * (
                                       global_best_position - particle.position)

            # Check if the updated velocity respect the velocity bounds
            for k in range(n):
                velocity_sign = np.sign(velocity_updated[k])
                velocity_updated[k] = velocity_sign * min(abs(velocity_updated[k]), max_velocity[k])

            position_updated = particle.position + velocity_updated

            # Linear decreasing inertia weight
            particle.w = (w_start - w_end) * (num_generations - (i + 1)) / num_generations + w_end

            particle.position = position_updated
            particle.velocity = velocity_updated

            # Handle integer variables
            particle.position_integer = np.around(particle.position)

            # Handle constraints using Death Penalty approach (non-feasible particles are not evaluated)
            if is_solution_feasible(particle.position_integer, bounds):
                if use_local_search:
                    # Memetic version of the algorithm
                    particle_value_updated = perform_memetic_variant(particle, n, bounds)

                    # Update global best according to the new value of the particle
                    if particle.best_value < global_best_value_i:
                        global_best_value_i = particle.best_value
                        global_best_position_i = particle.position_integer
                else:
                    particle_value_updated = particle.get_value()
                    n_function_evaluations += 1

                    # Updating best indexes
                    if particle_value_updated < particle.best_value:
                        particle.best_value = particle_value_updated
                        particle.best_position = np.array(particle.position_integer)

                    if particle_value_updated < global_best_value_i:
                        global_best_value_i = particle_value_updated
                        global_best_position_i = particle.position_integer

                logging.debug("New position : " + str(particle.position))
                logging.debug("New position (int) : " + str(particle.position_integer))
                logging.debug("New velocity : " + str(particle.velocity))
                logging.debug("Objective function value : " + str(particle_value_updated))
            else:
                logging.debug("New position : " + str(particle.position))
                logging.debug("New position (int) : " + str(particle.position_integer))
                logging.debug("New velocity : " + str(particle.velocity))
                logging.debug("Particle killed at this iteration")

        # Update best population indexes
        global_best_value = global_best_value_i
        global_best_position = np.array(global_best_position_i)

        logging.debug("\n\nEnd of generation")
        logging.debug("Best position : " + str(global_best_position))
        logging.debug("Best objective function value : " + str(global_best_value))

    logging.debug("\n\nTotal function evaluations: " + str(n_function_evaluations))
    return global_best_position, global_best_value


def is_solution_feasible(solution, bounds):
    for i in range(len(solution)):
        if solution[i] < bounds[i][0] or solution[i] > bounds[i][1]:
            return False

    return True


def perform_memetic_variant(particle, n, bounds):
    global n_function_evaluations

    # Evaluate the particle in the current integer position
    current_value = particle.get_value()
    n_function_evaluations += 1
    current_position = particle.position_integer
    # We keep track of the previous position to not evaluate the objective function there again
    previous_position = current_position
    best_value = current_value
    best_position = current_position

    logging.debug("\n --- Performing local search --- ")
    logging.debug(" Position : " + str(current_position))
    logging.debug(" With value : " + str(current_value))

    local_minimum_found = False
    while not local_minimum_found:
        # evaluate all 2*n integer positions adjacent to the current position
        for i in range(n):
            particle.position_integer = np.array(current_position)
            particle.position_integer[i] += 1
            if not np.array_equal(particle.position_integer, previous_position) and is_solution_feasible(particle.position_integer, bounds):
                value = particle.get_value()
                n_function_evaluations += 1

                if value < best_value:
                    best_value = value
                    best_position = particle.position_integer

            particle.position_integer = np.array(current_position)
            particle.position_integer[i] -= 1
            if not np.array_equal(particle.position_integer, previous_position) and is_solution_feasible(particle.position_integer, bounds):
                value = particle.get_value()
                n_function_evaluations += 1

                if value < best_value:
                    best_value = value
                    best_position = particle.position_integer

        if best_value == current_value:
            local_minimum_found = True
            logging.debug(" --- End local search --- \n")
        else:
            previous_position = current_position
            current_position = best_position
            current_value = best_value
            logging.debug(" Move to : " + str(current_position))
            logging.debug(" With value : " + str(current_value))

    if current_value < particle.best_value:
        # Move the particle into the new best personal position
        particle.position = np.array(current_position)
        particle.position_integer = np.array(current_position)
        particle.best_value = current_value
        particle.best_position = np.array(current_position)

    # Return the value associated to the current position of the particle
    return current_value
