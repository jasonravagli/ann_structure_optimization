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
        # min and max particle velocity used in random initialization
        self.min_init_velocity = [2] * n
        self.max_init_velocity = [5] * n
        # min and max particle velocity (if the particle velocity is below the lower bound, the particle is retired)
        # self.min_velocity = 1
        self.max_velocity = [5] * n
        self.initialization_type = InitializationType.QUASI_RANDOM
        # True = at each iteration if the particle has a non-integer position evaluates all the nearby integer positions
        self.use_local_search = False


def get_minimum(particle_factory, n, bounds, pso_hyperparameters=None):
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
    min_init_velocity = pso_hyperparameters.min_init_velocity
    max_init_velocity = pso_hyperparameters.max_init_velocity
    max_velocity = pso_hyperparameters.max_velocity
    initialization_type = pso_hyperparameters.initialization_type
    use_local_search = pso_hyperparameters.use_local_search

    particles = []
    best_position = None
    best_value = math.inf

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
        if particle_value < best_value:
            best_value = particle_value
            best_position = np.array(particle.position_integer)

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
        if particle_value < best_value:
            best_value = particle_value
            best_position = np.array(particle.position_integer)

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
        if particle_value < best_value:
            best_value = particle_value
            best_position = np.array(particle.position_integer)

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
        if particle_value < best_value:
            best_value = particle_value
            best_position = np.array(particle.position_integer)

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

        if particle_value < best_value:
            best_value = particle_value
            best_position = np.array(particle.position_integer)

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
        best_position_updated = best_position
        best_value_updated = best_value

        for j in range(swarm_size):
            logging.debug("\nParticle " + str(j))

            # Move particle
            particle = particles[j]

            # Particle with zero velocity are retired
            # if np.count_nonzero(particle.velocity) != 0:
            # Change the random hyperparameters
            r1 = np.random.rand()
            r2 = np.random.rand()

            # Particle swarm optimization update formulae
            velocity_updated = particle.w * particle.velocity + c1 * r1 * \
                               (particle.best_position - particle.position) + c2 * r2 * (
                                       best_position - particle.position)
            # Round the velocity array to integer values (we are solving a problem with integer variables)
            # velocity_updated = np.around(velocity_updated)

            # Check if the updated velocity respect the velocity bounds
            # if np.count_nonzero(velocity_updated) != 0:
            for k in range(n):
                velocity_sign = np.sign(velocity_updated[k])
                velocity_updated[k] = velocity_sign * min(abs(velocity_updated[k]), max_velocity[k])

            position_updated = particle.position + velocity_updated

            # Linear decreasing inertia weight
            # particle.w = w + 0.5 * math.exp(-LA.norm(particle.position - particle.best_position))
            particle.w = (w_start - w_end) * (num_generations - (i + 1)) / num_generations + w_end

            # Handle constraints
            # feasiblize_particle(particle, position_updated, velocity_updated, bounds)

            particle.position = position_updated
            particle.velocity = velocity_updated

            if use_local_search:
                particle_value_updated, best_value_updated, best_position_updated = local_search(particle, n, best_value_updated, best_position_updated)

                logging.debug("New position : " + str(particle.position))
                logging.debug("New position (int) : " + str(particle.position_integer))
                logging.debug("New velocity : " + str(particle.velocity))
                logging.debug("Objective function value : " + str(particle_value_updated))
            else:
                # Handle integer variables
                particle.position_integer = np.around(particle.position)

                # Handle constraints using Death Penalty approach (non-feasible particles are not evaluated)
                if is_solution_feasible(particle.position_integer, bounds):
                    particle_value_updated, best_value_updated, best_position_updated = evaluate_particle(particle, best_value_updated, best_position_updated)

                    # # Updating best indexes
                    # particle_value_updated = particle.get_value()
                    # if particle_value_updated < particle.best_value:  # and is_solution_feasible(particle.position, bounds):
                    #     particle.best_value = particle_value_updated
                    #     particle.best_position = np.array(particle.position_integer)
                    #
                    # if particle_value_updated < best_value_updated:
                    #     best_value_updated = particle_value_updated
                    #     best_position_updated = particle.position_integer

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
        best_value = best_value_updated
        best_position = np.array(best_position_updated)

        logging.debug("End of generation")
        logging.debug("Best position : " + str(best_position))
        logging.debug("Best objective function value : " + str(best_value))

    return best_position, best_value


def is_solution_feasible(solution, bounds):
    for i in range(len(solution)):
        if solution[i] < bounds[i][0] or solution[i] > bounds[i][1]:
            return False

    return True


def local_search(particle, n , best_value_updated, best_position_updated):
    # JR Il codice va generalizzato ma ho bisogno di fare delle prove immediate
    assert n == 2

    # Controllo il valore nelle 4 posizioni intere attorno alla posizione corrente della particella
    position_integer_0 = np.array([math.trunc(particle.position[0]), math.trunc(particle.position[1])])
    particle.position_integer = position_integer_0
    particle_value_updated_0, best_value_updated, best_position_updated = evaluate_particle(particle, best_value_updated, best_position_updated)

    position_integer_1 = np.array([math.trunc(particle.position[0]), math.ceil(particle.position[1])])
    particle.position_integer = position_integer_1
    particle_value_updated_1, best_value_updated, best_position_updated = evaluate_particle(particle, best_value_updated, best_position_updated)

    position_integer_2 = np.array([math.ceil(particle.position[0]), math.trunc(particle.position[1])])
    particle.position_integer = position_integer_2
    particle_value_updated_2, best_value_updated, best_position_updated = evaluate_particle(particle, best_value_updated, best_position_updated)

    position_integer_3 = np.array([math.ceil(particle.position[0]), math.ceil(particle.position[1])])
    particle.position_integer = position_integer_3
    particle_value_updated_3, best_value_updated, best_position_updated = evaluate_particle(particle, best_value_updated, best_position_updated)

    list_positions = [particle_value_updated_0, particle_value_updated_1, particle_value_updated_2, particle_value_updated_3]
    list_values = [particle_value_updated_0, particle_value_updated_1, particle_value_updated_2, particle_value_updated_3]
    particle_value_updated = min(list_values)
    particle.position_integer = list_positions[list_values.index(particle_value_updated)]
    return particle_value_updated, best_value_updated, best_position_updated


def evaluate_particle(particle, best_value_updated, best_position_updated):
    # Updating best indexes
    particle_value_updated = particle.get_value()
    if particle_value_updated < particle.best_value:
        particle.best_value = particle_value_updated
        particle.best_position = np.array(particle.position_integer)

    if particle_value_updated < best_value_updated:
        best_value_updated = particle_value_updated
        best_position_updated = particle.position_integer

    return particle_value_updated, best_value_updated, best_position_updated
