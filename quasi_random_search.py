import logging
import sobol_seq
import numpy as np
from AnnKFold import AnnKFold


def quasi_random_search(n_combinations, n_hyperparameters, bounds, x_train, y_train, x_test, y_test, n_fold):
    best_accuracy = 0
    best_params = None
    counter = 1

    for i in range(n_combinations):
        # Hyperparameters combinations generated by quasi random numbers with a Sobol sequence
        combination = np.array(sobol_seq.i4_sobol(n_hyperparameters, i+1)[0])

        for j in range(n_hyperparameters):
            # Put the position values into the bounds range
            combination[j] = bounds[j][0] + combination[j] * (bounds[j][1] - bounds[j][0])

        n_layers = int(np.around(combination[0]))
        n_neurons = int(np.around(combination[1]))

        print("\n\nCombination " + str(counter) + " out of " + str(n_combinations))
        logging.debug("\n\nCombination " + str(counter) + " out of " + str(n_combinations))
        logging.debug("N. layers : " + str(n_layers))
        logging.debug("N. neurons : " + str(n_neurons))

        ann = AnnKFold(n_fold)
        ann.x_train_set = x_train
        ann.y_train_set = y_train
        ann.x_test_set = x_test
        ann.y_test_set = y_test

        ann.create_model(n_layers, n_neurons, len(x_test[0]), len(y_test[0]))
        accuracy = ann.train_model()
        if accuracy > best_accuracy:
            best_params = [n_layers, n_neurons]
            best_accuracy = accuracy

        logging.debug("Accuracy on validation set : " + str(accuracy))
        counter += 1

    return best_params, best_accuracy