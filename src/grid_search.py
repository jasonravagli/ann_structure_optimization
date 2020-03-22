import logging
from sklearn.model_selection import ParameterGrid
from Ann import Ann
from AnnKFold import AnnKFold


def grid_search(grid, x_train, y_train, x_valid, y_valid, x_test, y_test):
    param_grid = {'layers': grid[0], 'neurons': grid[1]}

    param_grid = ParameterGrid(param_grid)

    best_accuracy = 0
    best_params = None
    for params in param_grid:
        ann = Ann()
        ann.x_train_set = x_train
        ann.y_train_set = y_train
        ann.x_valid_set = x_valid
        ann.y_valid_set = y_valid
        ann.x_test_set = x_test
        ann.y_test_set = y_test
        ann.create_model(params['layers'], params['neurons'], len(x_test[0]), len(y_test[0]))
        ann.train_model()
        accuracy = ann.evaluate_model()
        if accuracy > best_accuracy:
            best_params = [params['layers'], params['neurons']]
            best_accuracy = accuracy

    return best_params, best_accuracy


def grid_search_k_fold(grid, x_train, y_train, x_test, y_test, n_fold):
    param_grid = {'layers': grid[0], 'neurons': grid[1]}

    param_grid = ParameterGrid(param_grid)

    best_accuracy = 0
    best_params = None
    counter = 1
    tot_combinations = len(param_grid)
    for params in param_grid:
        print("\n\nCombination " + str(counter) + " out of " + str(tot_combinations))
        logging.debug("\n\nCombination " + str(counter) + " out of " + str(tot_combinations))
        logging.debug("N. layers : " + str(params['layers']))
        logging.debug("N. neurons : " + str(params['neurons']))

        ann = AnnKFold(n_fold)
        ann.x_train_set = x_train
        ann.y_train_set = y_train
        ann.x_test_set = x_test
        ann.y_test_set = y_test

        ann.create_model(params['layers'], params['neurons'], len(x_test[0]), len(y_test[0]))
        accuracy = ann.train_model()
        if accuracy > best_accuracy:
            best_params = [params['layers'], params['neurons']]
            best_accuracy = accuracy

        logging.debug("Accuracy on validation set : " + str(accuracy))
        counter += 1

    return best_params, best_accuracy
