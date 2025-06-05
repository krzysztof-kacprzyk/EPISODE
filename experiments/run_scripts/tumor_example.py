import sys
sys.path.append('..')
import experiments.benchmark as bench
from experiments.datasets import *
from episode.api import BSplineBasisFunctions, create_full_composition_library
from tqdm import tqdm

from episode.model_numpy import calculate_loss, calculate_loss_with_animation
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from io import StringIO

dataset = get_synthetic_tumor_dataset(500,20,0.01,0)

n_tune = 20
# experiment_name = args.name
max_epochs = 200
M=1
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
    'train_on_whole_trajectory': True,
}
basis_functions = [BSplineBasisFunctions(n_basis=6,k=3, include_linear=True, include_bias=False) for _ in range(M)]
n_motifs = 4
is_infinite = True
full_comp_library = create_full_composition_library(n_motifs,is_infinite=is_infinite, simplified=True)
composition_libraries = {m: create_full_composition_library(n_motifs,is_infinite=is_infinite, simplified=True) for m in range(M)}
config = {
    't_range': dataset.t_range,
    'M':1,
    'basis_functions':basis_functions,
    # 'composition_libraries':composition_libraries,
    'composition_libraries': {
            0: [
                ['++f'],
                ['-+c','++f'],
                ['-+h'],
                ['--c','-+h']
            ]
        },
    'opt_config':opt_config,
    'dt_config':dt_config,
    'x0_dict':dataset.x0,
    'verbose':True,
    'subtype':'biases_more'
}

psode_bench = bench.PSODEBenchmark(config, benchmarks_dir='results')

seed = 1826701614
train_indices, val_indices, test_indices = bench.generate_indices(len(dataset), 0.7, 0.15, seed = seed)
psode_bench.prepare_data(dataset, train_indices, val_indices, test_indices)
dataset_train_val = psode_bench.dataset_train_val
model = psode_bench.get_final_model(None, seed)
composition_scores = psode_bench._load_composition_scores(dataset.get_name(), seed, 0)
V, T, Y = dataset_train_val.get_V_T_Y()

composition_maps_dict = model.fit_composition_maps(V, T, Y, composition_scores_dict={0:composition_scores})

# Save the composition map
folder_path = "results/tumor_example"
file_name = "composition_map.txt"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

composition_maps_dict[0].print()

sys.stdout = old_stdout

composition_map_description = mystdout.getvalue()

composition_map_path = os.path.join(folder_path, file_name)
with open(composition_map_path, 'w') as f:
    for key, value in composition_maps_dict.items():
        f.write(composition_map_description)


model.fit(V,T,Y, composition_maps=composition_maps_dict)

property_submap = model.single_psodes[0].semantic_predictor.property_maps[('-+c','++f')]

property_submap.transition_point_predictor[(1,'x')].prune()

property_submap.transition_point_predictor[(1,'x')].visualize(max_n_columns=2)
# plt.subplots_adjust(bottom=-0.1)
plt.savefig(os.path.join(folder_path,'tumor_min_gam.pdf'), bbox_inches = 'tight')

property_submap.transition_point_predictor[(1,'x')].visualize(max_n_columns=2)
# plt.subplots_adjust(bottom=-0.1)
plt.savefig(os.path.join(folder_path,'tumor_min_gam.svg'), bbox_inches = 'tight')

gam = property_submap.transition_point_predictor[(1,'x')]
bias = gam.bias * 1.0
for i in range(gam.n_shape_functions):
    bias += gam.shape_functions[i].get_expected_value()

# Save the bias
bias_file_path = os.path.join(folder_path, 'bias.txt')
with open(bias_file_path, 'w') as f:
    f.write(str(bias))

