import math


class ParticleAnnKFold:
    def __init__(self, annKFold):
        self.position = None
        self.velocity = None
        self.position_integer = None
        self.best_position = None  # position of the best value encountered by the particle
        self.best_value = math.inf  # best value of the objective function encountered by the particle
        self.w = 0  # dynamic inertia weight to use in PSO modified formula
        self.annKFold = annKFold  # neural network to optimize

    def get_value(self):
        """
        Create the ANN model with the current hyperparameters (position), train it and return its performance on the validation set
        :return: Network accuracy on the validation set
        """

        n_layers = int(self.position_integer[0])
        n_neurons = int(self.position_integer[1])

        self.annKFold.create_model(n_layers, n_neurons, len(self.annKFold.x_test_set[0]), len(self.annKFold.y_test_set[0]))

        return 1 - self.annKFold.train_model()
