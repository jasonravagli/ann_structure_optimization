from ParticleAnnKFold import ParticleAnnKFold
from AnnKFold import AnnKFold


class ParticleAnnKFoldFactory:

    def __init__(self, x_train, y_train, x_test, y_test, n_fold):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_fold = n_fold

    def get_particle(self):
        ann = AnnKFold(self.n_fold)
        ann.x_train_set = self.x_train
        ann.y_train_set = self.y_train
        ann.x_test_set = self.x_test
        ann.y_test_set = self.y_test

        particle = ParticleAnnKFold(ann)
        return particle
