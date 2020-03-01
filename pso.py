import random
import sobol_seq
import math
import numpy as np
import logging
from numpy import linalg as LA


# PSO Hyperparameters
class PSOHyperparameters:

    def __init__(self, n):
        """
        :param n: Dimension of the problem to be resolved by te pso algorithm
        """
        self.w = 0.5
        self.c1 = 0.5
        self.c2 = 0.5
        self.swarm_size = 10
        self.num_generations = 10
        # min and max particle velocity used in random initialization
        self.min_init_velocity = [2] * n
        self.max_init_velocity = [5] * n
        # min and max particle velocity (if the particle velocity is below the lower bound, the particle is retired)
        #self.min_velocity = 1
        self.max_velocity = [5] * n


def get_minimum(particle_factory, n, bounds, pso_hyperparameters=None):
    # Number of bounds must be equal to the problem dimension (number of problem hyperparameters)
    assert len(bounds) == n

    if pso_hyperparameters is None:
        pso_hyperparameters = PSOHyperparameters(n)

    w = pso_hyperparameters.w
    c1 = pso_hyperparameters.c1
    c2 = pso_hyperparameters.c2
    swarm_size = pso_hyperparameters.swarm_size
    num_generations = pso_hyperparameters.num_generations
    min_init_velocity = pso_hyperparameters.min_init_velocity
    max_init_velocity = pso_hyperparameters.max_init_velocity
    max_velocity = pso_hyperparameters.max_velocity

    particles = []
    best_position = None
    best_value = math.inf

    # Random initialize the particles and evaluate them
    print("\n\n***** Particles initialization *****")
    logging.debug("\n\n***** Particles initialization *****")
    for i in range(swarm_size):
        # position = np.zeros(n)
        velocity = np.zeros(n)

        # Initial position is formed by quasi random numbers given by a Sobol sequence
        position = np.array(sobol_seq.i4_sobol(n, i+1)[0])

        for j in range(n):
            # Put the position values into the bounds range
            position[j] = bounds[j][0] + position[j] * (bounds[j][1] - bounds[j][0])

            # Select random velocity within defined bounds and add a random sign to it
            velocity[j] = np.random.choice([-1, 1]) * np.random.uniform(min_init_velocity[j], max_init_velocity[j] + 1)

        # for j in range(n):
            # position[j] = np.random.randint(bounds[j][0], bounds[j][1] + 1)
            # Select random velocity within defined bounds and add a random sign to it
            # velocity[j] = np.random.choice([-1, 1]) * np.random.randint(min_init_velocity[j], max_init_velocity[j] + 1)

        particle = particle_factory.get_particle()
        particle.position = position
        particle.position_integer = np.around(position)
        particle.velocity = velocity
        particle.w = w

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
            #if np.count_nonzero(particle.velocity) != 0:
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

            # Update dynamic inertia weight associated to the particle (PSO variant)
            particle.w = w + 0.5 * math.exp(-LA.norm(particle.position - particle.best_position))

            # Handle constraints
            feasiblize_particle(particle, position_updated, velocity_updated, bounds)
            particle.position_integer = np.around(particle.position)

            # Updating best indexes
            particle_value_updated = particle.get_value()
            if particle_value_updated < particle.best_value: #and is_solution_feasible(particle.position, bounds):
                particle.best_value = particle_value_updated
                particle.best_position = np.array(particle.position_integer)

            if particle_value_updated < best_value_updated:
                best_value_updated = particle_value_updated
                best_position_updated = particle.position_integer

            logging.debug("New position : " + str(particle.position))
            logging.debug("New position (int) : " + str(particle.position_integer))
            logging.debug("New velocity : " + str(particle.velocity))
            logging.debug("Objective function value : " + str(particle_value_updated))
            # else:
            #     logging.debug("Particle retired")

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


def feasiblize_particle(particle, position_updated, velocity_updated, bounds):
    #  Constraint handling method: bounds form an hypercube and particles hit the hypercube sides bouncing on them

    for i in range(len(particle.position)):
        if position_updated[i] < bounds[i][0]:
            position_updated[i] = bounds[i][0] + (bounds[i][0] - position_updated[i])
            # Elastic impact: the velocity is preserved but changed in sign
            velocity_updated[i] = -velocity_updated[i]
        elif position_updated[i] > bounds[i][1]:
            position_updated[i] = bounds[i][1] - (position_updated[i] - bounds[i][1])
            # Elastic impact: the velocity is preserved but changed in sign
            velocity_updated[i] = -velocity_updated[i]

    particle.position = position_updated
    particle.velocity = velocity_updated
