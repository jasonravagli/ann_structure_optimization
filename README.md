# ANN structure optimizazion using PSO algorithm
In this Python project different neural network hyperparameters optimization methods are used to optimize the
architecture of an MLP neural network used in a binary classification problem.

The methods optimizes the number of layers and the number of neurons in each layer (supposing that all layers have the
same number of neurons) that compose the network.

In particular, the project focuses on the implementation of the Particle Swarm Optimization algorithm,
a global optimization algorithm, and the comparison between this one and the other implementd methods.

In the project three different optimization methods are implemented:
* Quasi-Random
* Grid Search
* PSO

For the PSO are also provided three variants (please refer to the PDF report included in the repository for further details):
* __pso__: base implementation
* __pso_init_borders__: a variant where the particles are initialized also at the borders of the feasible hyperparameters space
* __pso_local_search__: a variant where the objective function evaluations are made for all the integer position around the particles current position

For the binary classification problem, we use the Diabetic Retinopathy Debrecen dataset
(https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set), already included in the project
as .arff file.


## System Requirements
* Python 3.7 or later
* Keras 2.3.1 or later and its dependencies (e.g. Tensorflow)
* sobol-seq 0.1.2 or later

## Usage
From the command line:
* Download the project: `git clone https://github.com/jasonravagli/ann_structure_optimization.git`
* Enter in the source folder `cd ann_structure_optimization/src`
* Run the main.py file: `python main.py`

You can specify two arguments to the main.py file:
* `--method` optimization method to use.

    Possible values: `all, grid_search, quasi_random, pso (default), pso_init_borders, pso_local_search`
* `--n_exp` number of different experiments to perform for each selected method

## Output
The output files are saved in the `/results_*` directories. One directory for each used method will be created.
