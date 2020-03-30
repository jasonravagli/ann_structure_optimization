import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import pso
import dataset_reader
import time
import logging
import grid_search
import quasi_random_search
from ParticleAnnFactory import ParticleAnnFactory
from ParticleAnnKFoldFactory import ParticleAnnKFoldFactory
from Ann import Ann


def pso_optimization(generate_validation_set, n_fold, x_train, y_train, x_valid, y_valid, x_test, y_test, initialization_type=pso.InitializationType.QUASI_RANDOM, use_local_search=False):
    print("\nReading dataset...")

    print("\n**** Dataset statistics *****")
    print("Training samples: " + str(len(x_train)))
    if generate_validation_set:
        print("Validation samples: " + str(len(x_valid)))
    print("Test samples: " + str(len(x_test)))

    if generate_validation_set:
        particleFactory = ParticleAnnFactory(x_train, y_train, x_valid, y_valid, x_test, y_test)
    else:
        particleFactory = ParticleAnnKFoldFactory(x_train, y_train, x_test, y_test, n_fold)

    n = 2  # Problem dimension (hyperparameters to be tuned)

    # *** Setting PSO algorithm hyperparameters

    layers_bounds = (1, 10)
    neurons_bounds = (4, 384)

    pso_hyperparameters = pso.PSOHyperparameters(n)
    pso_hyperparameters.w_start = 0.9
    pso_hyperparameters.w_end = 0.4
    pso_hyperparameters.c1 = 0.5
    pso_hyperparameters.c2 = 0.5
    pso_hyperparameters.swarm_size = 10
    pso_hyperparameters.num_generations = 10
    pso_hyperparameters.max_velocity = [3, 32]
    pso_hyperparameters.initialization_type = initialization_type
    pso_hyperparameters.use_local_search = use_local_search

    logging.info("\n\n***** PSO Configuration ******")
    logging.info("w_start : " + str(pso_hyperparameters.w_start))
    logging.info("w_end : " + str(pso_hyperparameters.w_end))
    logging.info("c1 : " + str(pso_hyperparameters.c1))
    logging.info("c2 : " + str(pso_hyperparameters.c2))
    logging.info("swarm_size : " + str(pso_hyperparameters.swarm_size))
    logging.info("num_generations : " + str(pso_hyperparameters.num_generations))
    logging.info("max_velocity : " + str(pso_hyperparameters.max_velocity))
    logging.info("bounds : " + str([layers_bounds, neurons_bounds]))
    logging.info("initialization type : " + str(pso_hyperparameters.initialization_type))
    logging.info("use local search : " + str(pso_hyperparameters.use_local_search))

    start = time.time()
    min_point, min_value = pso.get_minimum(particleFactory, n, [layers_bounds, neurons_bounds], pso_hyperparameters)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\nMinimum point: " + str(min_point))
    print("Minimum value: " + str(min_value))
    print("Execution time : " + ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))
    logging.info("\n\n***** Optimal configuration found by PSO ******")
    logging.info("N. hidden layers : " + str(min_point[0]))
    logging.info("N. neurons per layer : " + str(min_point[1]))
    logging.info("Accuracy on validation set : " + str(1 - min_value))
    logging.info("Execution time : " + ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))

    # With the optimal structure found retrain the network and calculate accuracy on test set
    n_layers = int(min_point[0])
    n_neurons = int(min_point[1])

    if generate_validation_set:
        ann = Ann()
        ann.x_train_set = x_train
        ann.y_train_set = y_train
        ann.x_valid_set = x_valid
        ann.y_valid_set = y_valid
        ann.x_test_set = x_test
        ann.y_test_set = y_test
        ann.create_model(n_layers, n_neurons, len(ann.x_test_set[0]), len(ann.y_test_set[0]))
    else:
        ann = Ann()
        ann.x_train_set = x_train
        ann.y_train_set = keras.utils.to_categorical(y_train, 2)
        ann.x_test_set = x_test
        ann.y_test_set = y_test
        ann.create_model(n_layers, n_neurons, len(ann.x_test_set[0]), len(ann.y_test_set[0]))
        validation_split = 0.25
        ann.train_model(validation_split)

    accuracy = ann.evaluate_model()

    print("\nAccuracy with " + str(n_layers) + " layers and " + str(n_neurons) + " neurons: " + str(accuracy))
    logging.info("Accuracy on test set : " + str(accuracy))


def grid_search_optimization(generate_validation_set, n_fold, x_train, y_train, x_valid, y_valid, x_test, y_test):
    grid = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            # [4, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384]]
            [4, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]]

    print("\n**** Dataset statistics *****")
    print("Training samples: " + str(len(x_train)))
    if generate_validation_set:
        print("Validation samples: " + str(len(x_valid)))
    print("Test samples: " + str(len(x_test)))

    logging.info("***** Grid Search configuration *****")
    logging.info("Grid: " + str(grid))

    start = time.time()
    if generate_validation_set:
        min_point, min_value = grid_search.grid_search(grid, x_train, y_train, x_valid, y_valid, x_test, y_test)
    else:
        min_point, min_value = grid_search.grid_search_k_fold(grid, x_train, y_train, x_test, y_test, n_fold)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\nMinimum point: " + str(min_point))
    print("Minimum value: " + str(min_value))
    print("Execution time : " + ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))
    logging.info("\n\n***** Optimal configuration found by Grid Search ******")
    logging.info("N. hidden layers : " + str(min_point[0]))
    logging.info("N. neurons per layer : " + str(min_point[1]))
    logging.info("Accuracy on validation set : " + str(min_value))
    logging.info("Execution time : " + ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))

    # With the optimal structure found retrain the network and calculate accuracy on test set
    n_layers = int(min_point[0])
    n_neurons = int(min_point[1])

    if generate_validation_set:
        ann = Ann()
        ann.x_train_set = x_train
        ann.y_train_set = y_train
        ann.x_valid_set = x_valid
        ann.y_valid_set = y_valid
        ann.x_test_set = x_test
        ann.y_test_set = y_test
        ann.create_model(n_layers, n_neurons, len(ann.x_test_set[0]), len(ann.y_test_set[0]))
    else:
        ann = Ann()
        ann.x_train_set = x_train
        ann.y_train_set = keras.utils.to_categorical(y_train, 2)
        ann.x_test_set = x_test
        ann.y_test_set = y_test
        ann.create_model(n_layers, n_neurons, len(ann.x_test_set[0]), len(ann.y_test_set[0]))
        validation_split = 0.25
        ann.train_model(validation_split)

    accuracy = ann.evaluate_model()

    print("\nAccuracy with " + str(n_layers) + " layers and " + str(n_neurons) + " neurons: " + str(accuracy))
    logging.info("Accuracy on test set : " + str(accuracy))


def quasi_random_optimization(generate_validation_set, n_fold, x_train, y_train, x_valid, y_valid, x_test, y_test):

    print("\n**** Dataset statistics *****")
    print("Training samples: " + str(len(x_train)))
    if generate_validation_set:
        print("Validation samples: " + str(len(x_valid)))
    print("Test samples: " + str(len(x_test)))

    n = 2  # Problem dimension (hyperparameters to be tuned)

    n_combinations = 10
    layers_bounds = (1, 10)
    neurons_bounds = (4, 384)

    logging.info("\n\n***** Quasi-Random Search Configuration ******")
    logging.info("combinations : " + str(n_combinations))
    logging.info("bounds : " + str([layers_bounds, neurons_bounds]))

    start = time.time()
    min_point, min_value = quasi_random_search.quasi_random_search(n_combinations, n, [layers_bounds, neurons_bounds], x_train, y_train, x_test, y_test, n_fold)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\nMinimum point: " + str(min_point))
    print("Minimum value: " + str(min_value))
    print("Execution time : " + ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))
    logging.info("\n\n***** Optimal configuration found by Quasi-Random Search ******")
    logging.info("N. hidden layers : " + str(min_point[0]))
    logging.info("N. neurons per layer : " + str(min_point[1]))
    logging.info("Accuracy on validation set : " + str(1 - min_value))
    logging.info("Execution time : " + ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))

    # With the optimal structure found retrain the network and calculate accuracy on test set
    n_layers = int(min_point[0])
    n_neurons = int(min_point[1])

    ann = Ann()
    ann.x_train_set = x_train
    ann.y_train_set = keras.utils.to_categorical(y_train, 2)
    ann.x_test_set = x_test
    ann.y_test_set = y_test
    ann.create_model(n_layers, n_neurons, len(ann.x_test_set[0]), len(ann.y_test_set[0]))
    validation_split = 0.25
    ann.train_model(validation_split)

    accuracy = ann.evaluate_model()

    print("\nAccuracy with " + str(n_layers) + " layers and " + str(n_neurons) + " neurons: " + str(accuracy))
    logging.info("Accuracy on test set : " + str(accuracy))
