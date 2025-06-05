import sys
sys.path.append('..')
from experiments.datasets import *
from experiments.utils import *
from experiments.benchmark import PSODEBenchmark, run_benchmark_random_splits
from episode.api import BSplineBasisFunctions, create_full_composition_library

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import torch

n_trials = 1
global_seed = 0
run_seed = 1826701614
n_tune = 20
experiment_name = "SIR_plots"
n_samples = 500
max_epochs = 200
n_measurements = 20
noise_std = 0.01

dataset_config = {
    'n_samples': n_samples,
    'n_measurements': n_measurements,
    'noise_std': noise_std,
    'seed': global_seed
}

dataset = get_SIR_dataset(**dataset_config)
M = dataset.get_M()

opt_config = {
        'lr': 0.1,
        'n_epochs': max_epochs,
        'batch_size': 1024,
        'weight_decay': 0.0,
        'device': 'cpu',
        'dis_loss_coeff_1': 1e-2,
        'dis_loss_coeff_2': 1e-6,
        'last_loss_coeff': 100.0,
        'n_tune':n_tune,
        'dtw':False
    }
dt_config = {
    'max_depth': 3,
    'min_relative_gain_to_split': 1e-2,
    'min_samples_leaf':50,
    'relative_motif_cost': 1e-2,
    'tune_depth': True,
    'train_on_whole_trajectory': True, # whether to train model_numpy on the whole trajectory or just the first 80%
}
basis_functions = [BSplineBasisFunctions(n_basis=6,k=3, include_linear=True, include_bias=False) for _ in range(M)]

composition_libraries = get_inductive_bias_composition_libraries("SIR")
subtype = "biases_more"

config = {
    't_range': dataset.t_range,
    'M':M,
    'basis_functions':basis_functions,
    'composition_libraries':composition_libraries,
    'opt_config':opt_config,
    'dt_config':dt_config,
    'x0_dict':dataset.x0,
    'verbose':True,
    'subtype': subtype,
}

baseline = PSODEBenchmark(config)
results, model = run_benchmark_random_splits(dataset, baseline, n_trials=n_trials, n_tune=0, seed=run_seed, experiment_name=experiment_name, benchmarks_dir="results")

subset, _ = dataset.split(0.1)
new_T = np.stack([np.linspace(0,1,1000) for i in range(len(subset))], axis=0)
from episode.data import Dataset
subset_modified = Dataset(subset.name, subset.V, new_T, subset.Y, subset.t_range, subset.x0)
Y_pred = baseline.predict(model, subset_modified)

Y_true = subset.get_Y()
T = subset.get_T()

import matplotlib.pyplot as plt

chosen_indices = [0,1]

for number, index in enumerate(chosen_indices):

    plt.figure()

    plt.scatter(T[index], Y_true[index][:,0], label='S (observed)')
    plt.plot(new_T[index], Y_pred[index][:,0], label='S (predicted)')
    plt.scatter(T[index], Y_true[index][:,1], label='I (observed)')
    plt.plot(new_T[index], Y_pred[index][:,1], label='I (predicted)')
    plt.scatter(T[index], Y_true[index][:,2], label='R (observed)')
    plt.plot(new_T[index], Y_pred[index][:,2], label='R (predicted)')

    plt.legend()

    plt.savefig(f"results/SIR_plots/SIR_{number}.png", bbox_inches = 'tight')
    plt.savefig(f"results/SIR_plots/SIR_{number}.svg", bbox_inches = 'tight')
    plt.savefig(f"results/SIR_plots/SIR_{number}.pdf", bbox_inches = 'tight')