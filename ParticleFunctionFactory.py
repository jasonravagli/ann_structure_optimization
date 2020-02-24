from ParticleFunction import ParticleFunction


class ParticleFunctionFactory:

    def __init__(self, f):
        self.function = f

    def get_particle(self):
        particle = ParticleFunction(self.function)
        return particle
