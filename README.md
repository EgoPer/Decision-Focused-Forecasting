# Decision Focused Forecasting
This repository contains the code to replicate the experiment in the eponymous paper.

The data for the experiment (contained in *Data/power_scheduling*) was copied from the [respository](https://github.com/locuslab/e2e-model-learning) associated with Donti et al. (2017) which has an Apache 2.0 licence. We also borrow some of the data loading code in *data_utils.py*.

## Running 

The models are built with PyTorch and cvxpylayers, implementation is in *models.py*. We provide a *requirements.py* file to clarify other dependencies. To try the experiment with diffferent hyperparameters the *hyperparameters.py* file contains a function in which settings may be changed. To run the experiment call *run.py* in which one can define how many replications to run. The file runs the experiment function defined in *experiment_whole_policy_evaluation.py* which contains the whole pipeline as assembled from other files.

## Other files
- *Results/policy/* contains the results of the experiments that we ran including saved models. 
- *data_utils.py* contains loading/processing functions for data and dataset classes for training and evaluation.
- *evaluation_utils.py* contains evaluation functions.
- *optimisation_utils.py* contains functions which parametrically build the cvxpy problem and optimisation layer, and the torch loss function for the DFF optimisation.
