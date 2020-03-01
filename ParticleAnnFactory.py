from ParticleAnn import ParticleAnn
from Ann import Ann


class ParticleAnnFactory:

    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

    def get_particle(self):
        ann = Ann()
        ann.x_train_set = self.x_train
        ann.y_train_set = self.y_train
        ann.x_valid_set = self.x_valid
        ann.y_valid_set = self.y_valid
        ann.x_test_set = self.x_test
        ann.y_test_set = self.y_test

        particle = ParticleAnn(ann)
        return particle
