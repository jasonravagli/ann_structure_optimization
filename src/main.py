import argparse
import pso
import dataset_reader
import optimization_functions
import time
import logging
import shutil
from pathlib import Path

# Configuring logging to file
directory_output = "../output"
Path(directory_output).mkdir(parents=True, exist_ok=True)
log_filename = '../output/output.log'
logging.basicConfig(filename=log_filename, format='%(message)s', level=logging.DEBUG)


if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser(description='ANN Structure Optimization')
    parser.add_argument("--method",
                        choices=["all", "pso", "pso_init_borders", "pso_local_search", "grid_search", "quasi_random"],
                        default="pso", type=str, help="Optimization method to use")
    parser.add_argument("--n_exp", default=1, type=int, help="Number of experiments to perform for each selected method")

    args = parser.parse_args()

    method = args.method
    n_experiments = args.n_exp

    # False = use K-Fold cross validation during training; True = use fixed validation set
    generate_validation_set = False
    n_fold = 5
    preprocessing = dataset_reader.DatasetPreprocessing.NORMALIZE

    # Create output directories
    directory_grid_search = "../results_grid"
    directory_pso = "../results_pso"
    directory_pso_border_init = "../results_pso_border_init"
    directory_pso_local_search = "../results_pso_local_search"
    directory_quasi_random = "../results_quasi_random"
    Path(directory_grid_search).mkdir(parents=True, exist_ok=True)
    Path(directory_pso).mkdir(parents=True, exist_ok=True)
    Path(directory_pso_border_init).mkdir(parents=True, exist_ok=True)
    Path(directory_pso_local_search).mkdir(parents=True, exist_ok=True)
    Path(directory_quasi_random).mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset_reader.read_diabetic_retinopathy_debrecen(n_fold,
                                                                                                           generate_validation_set, preprocessing)

    for i in range(n_experiments):
        print("\n\n****** Experiment " + str(i+1) + " out of " + str(n_experiments) + " *******")

        if method == 'all' or method == 'grid_search':
            print("\n\n****** Grid Search Method ******")
            open(log_filename, 'w').close()  # Empty log file
            optimization_functions.grid_search_optimization(generate_validation_set, n_fold, x_train, y_train, x_valid, y_valid, x_test, y_test)
            shutil.copy(log_filename, directory_grid_search + "/" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".log")

        if method == 'all' or method == 'pso':
            print("\n\n****** PSO Method ******")
            open(log_filename, 'w').close()
            optimization_functions.pso_optimization(generate_validation_set, n_fold, x_train, y_train, x_valid, y_valid, x_test, y_test)
            shutil.copy(log_filename, directory_pso + "/" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".log")

        if method == 'all' or method == 'pso_init_borders':
            print("\n\n****** PSO (Init on Borders) Method ******")
            open(log_filename, 'w').close()
            optimization_functions.pso_optimization(generate_validation_set, n_fold, x_train, y_train, x_valid, y_valid, x_test, y_test, initialization_type=pso.InitializationType.QUASI_RANDOM_USING_BORDER)
            shutil.copy(log_filename, directory_pso_border_init + "/" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".log")

        if method == 'all' or method == 'pso_local_search':
            print("\n\n****** PSO Method (Memetic Variant) ******")
            open(log_filename, 'w').close()
            optimization_functions.pso_optimization(generate_validation_set, n_fold, x_train, y_train, x_valid, y_valid, x_test, y_test, use_local_search=True)
            shutil.copy(log_filename, directory_pso_local_search + "/" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".log")

        if method == 'all' or method == 'quasi_random':
            print("\n\n****** Quasi-Random Method ******")
            open(log_filename, 'w').close()
            optimization_functions.quasi_random_optimization(generate_validation_set, n_fold, x_train, y_train, x_valid, y_valid, x_test, y_test)
            shutil.copy(log_filename, directory_quasi_random + "/" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".log")
