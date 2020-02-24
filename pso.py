import random
import math
import numpy as np
from numpy import linalg as LA

# PSO Hyperparameters
w = 0.5
c1 = 0.5
c2 = 0.5
swarm_size = 10
num_generations = 5
# min and max particle velocity used in random initialization
min_init_velocity = 2
max_init_velocity = 5
# min and max particle velocity (if the particle velocity is below the lower bound, the particle is retired)
# min_velocity = 1 --> implicit
max_velocity = 5


def get_minimum(particle_factory, n, bounds):
    global w, c1, c2, swarm_size, num_generations, r1, r2, min_init_velocity, max_init_velocity

    # Number of bounds must be equal to the problem dimension (number of hyperparameters)
    assert len(bounds) == n

    particles = []
    best_position = None
    best_value = math.inf

    # Random initialize the particles and evaluate them
    for i in range(swarm_size):
        position = np.zeros(n)
        for j in range(n):
            position[j] = np.random.randint(bounds[j][0], bounds[j][1] + 1)

        particle = particle_factory.get_particle()
        particle.position = position
        particle.velocity = np.random.randint(min_init_velocity, max_init_velocity + 1, n)
        particle.w = w

        particle_value = particle.get_value()

        particle.best_position = np.array(position)
        particle.best_value = particle_value

        if particle_value < best_value:
            best_value = particle_value
            best_position = np.array(particle.position)

        print("Particle " + str(i) + " - Position " + str(particle.position))

        particles.append(particle)

    # PSO Optimization
    for i in range(num_generations):
        # Auxiliary variables to temporary store best population indexes (the update of the best indexes is done after the particles update)
        best_position_updated = best_position
        best_value_updated = best_value

        for j in range(swarm_size):
            # Move particle
            particle = particles[j]

            # Particle with zero velocity are retired
            if np.count_nonzero(particle.velocity) != 0:
                # Change the random hyperparameters
                r1 = np.random.rand()
                r2 = np.random.rand()

                # Particle swarm optimization update formulae
                velocity_updated = particle.w * particle.velocity + c1 * r1 * \
                                   (particle.best_position - particle.position) + c2 * r2 * (
                                           best_position - particle.position)
                # Round the velocity array to integer values (we are solving a problem with integer variables)
                velocity_updated = np.around(velocity_updated)

                # Check if the updated velocity respect the velocity bounds
                if np.count_nonzero(velocity_updated) != 0:
                    for k in range(n):
                        velocity_sign = np.sign(velocity_updated[k])
                        velocity_updated[k] = velocity_sign * min(abs(velocity_updated[k]), max_velocity)

                    position_updated = particle.position + velocity_updated

                    # Update dynamic inertia weight associated to the particle (PSO variant)
                    particle.w = w + 0.5 * math.exp(-LA.norm(particle.position - particle.best_position))

                    # Handle constraints
                    feasiblize_particle(particle, position_updated, velocity_updated, bounds)

                    # Updating best indexes
                    particle_value_updated = particle.get_value()
                    if particle_value_updated < particle.best_value and is_solution_feasible(particle.position, bounds):
                        particle.best_value = particle_value_updated
                        particle.best_position = np.array(particle.position)

                    if particle_value_updated < best_value_updated:
                        best_value_updated = particle_value_updated
                        best_position_updated = particle.position

        # Update best population indexes
        best_value = best_value_updated
        best_position = np.array(best_position_updated)

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
