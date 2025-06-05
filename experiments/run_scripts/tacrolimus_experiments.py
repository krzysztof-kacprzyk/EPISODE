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

import torch


folder_path = os.path.join('results','tacolimus')

def save_results(method, mean, std):

    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path,'summary.csv')

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=['method','mean','std'])
    else:
        df = pd.read_csv(file_path)

    new_row = pd.DataFrame({'method':[method],'mean':[mean],'std':[std]})
    df = pd.concat([df,new_row],ignore_index=True)

    df.to_csv(file_path, index=False)

dataset = get_real_pharma_dataset()

opt_config = {
        'lr': 0.1,
        'n_epochs': 200,
        'batch_size': 1024,
        'weight_decay': 0.0,
        'device': 'cpu',
        'dis_loss_coeff_1': 1e-2,
        'dis_loss_coeff_2': 1e-6,
        'last_loss_coeff': 100.0,
        # 'n_tune':20
        'n_tune':20,
        'dtw':False
}
dt_config = {
    'max_depth': 3,
    'min_relative_gain_to_split': 1e-2,
    'min_samples_leaf':50,
    'relative_motif_cost': 1e-2,
    'tune_depth': True,
}
# dt_config = {
#     'max_depth': 3,
#     'min_relative_gain_to_split': 0,
#     'min_samples_leaf':20,
#     'relative_motif_cost': 0,
#     'tune_depth': False,
# }
basis_functions = [BSplineBasisFunctions(n_basis=6,k=3, include_linear=True, include_bias=False) for _ in range(1)]

composition_libraries = {
    0: [
    ['+-c', '--c', '-+h'],
    ['--c', '-+h'],
    ['-+h'],
    ['++c', '+-c', '--c', '-+h']
    ]
}
config = {
    't_range': dataset.t_range,
    'M':1,
    'basis_functions':basis_functions,
    'composition_libraries':composition_libraries,
    'opt_config':opt_config,
    'dt_config':dt_config,
    'x0_dict':dataset.x0,
    'verbose':True,
    'subtype':'more'
}

psode_bench = bench.PSODEBenchmark(config, benchmarks_dir='results')

seed = 1826701614
train_indices, val_indices, test_indices = bench.generate_indices(len(dataset), 0.7, 0.15, seed = seed)
psode_bench.prepare_data(dataset, train_indices, val_indices, test_indices)
dataset_train_val = psode_bench.dataset_train_val
model = psode_bench.get_final_model(None, seed)
composition_scores = psode_bench._load_composition_scores(dataset.get_name(), seed, 0)
composition_scores = composition_scores[:,[4,9,14,25]]
V, T, Y = dataset_train_val.get_V_T_Y()

composition_maps_dict = model.fit_composition_maps(V, T, Y, composition_scores_dict={0:composition_scores})

model.fit(V,T,Y,composition_maps_dict)

mean, std = psode_bench.evaluate_on_test(model)

print("Performance of PSODE on test set: ", mean, std)
save_results('PSODE', mean, std)

model.single_psodes[0].semantic_predictor.property_maps[('++c','+-c','--c','-+h')].infinite_motif_predictor[0].visualize(max_n_columns=5)

plt.savefig(os.path.join(folder_path,'tacrolimus_h_original.pdf'), bbox_inches = 'tight')
plt.savefig(os.path.join(folder_path,'tacrolimus_h_original.svg'), bbox_inches = 'tight')


def zero_asymptote(finite_coordinates,derivative,properties):
    loss = torch.mean(properties[:,0] ** 2) * 1 #+ torch.mean(torch.nn.functional.relu(properties[:,1] - 6.0)) * 1
    return loss

soft_constraints = {
    ('++c','+-c','--c','-+h'): zero_asymptote
}
model = psode_bench.get_final_model(None, seed)
model.fit(V,T,Y,composition_maps_dict, all_soft_constraints={0:soft_constraints})

mean, std = psode_bench.evaluate_on_test(model)
print("Performance of PSODE on test set: ", mean, std)
save_results('PSODE-zero', mean, std)

model.single_psodes[0].semantic_predictor.property_maps[('++c','+-c','--c','-+h')].infinite_motif_predictor[0].visualize(max_n_columns=5)
plt.savefig(os.path.join(folder_path,'tacrolimus_h_0_loss.pdf'), bbox_inches = 'tight')
model.single_psodes[0].semantic_predictor.property_maps[('++c','+-c','--c','-+h')].infinite_motif_predictor[0].visualize(max_n_columns=5)
plt.savefig(os.path.join(folder_path,'tacrolimus_h_0_loss.svg'), bbox_inches = 'tight')

property_map = model.single_psodes[0].semantic_predictor.property_maps[('++c','+-c','--c','-+h')]
n_shapes = property_map.infinite_motif_predictor[0].n_shape_functions
for i in range(n_shapes):
    property_map.infinite_motif_predictor[0].shape_functions[i].function = lambda x: np.zeros_like(x)

property_map.infinite_motif_predictor[0].bias = 0

property_map.infinite_motif_predictor[0].visualize(max_n_columns=5)
plt.savefig(os.path.join(folder_path,'tacrolimus_h_0_loss_zeroed.pdf'), bbox_inches = 'tight')
property_map.infinite_motif_predictor[0].visualize(max_n_columns=5)
plt.savefig(os.path.join(folder_path,'tacrolimus_h_0_loss_zeroed.svg'), bbox_inches = 'tight')

mean, std = psode_bench.evaluate_on_test(model)
print("Performance of PSODE on test set: ", mean, std)
save_results('PSODE-verified', mean, std)

property_map.transition_point_predictor[(2,'t')].prune(1e-5)
property_map.transition_point_predictor[(2,'t')].visualize(max_n_columns=5, skip_constant=True)
plt.savefig(os.path.join(folder_path,'tacrolimus_max_t_pruned.pdf'), bbox_inches = 'tight')
property_map.transition_point_predictor[(2,'t')].visualize(max_n_columns=5, skip_constant=True)
plt.savefig(os.path.join(folder_path,'tacrolimus_max_t_pruned.svg'), bbox_inches = 'tight')


property_map.transition_point_predictor[(2,'x')].prune(1e-5)
property_map.transition_point_predictor[(2,'x')].visualize(max_n_columns=4, skip_constant=True)
plt.savefig(os.path.join(folder_path,'tacrolimus_max_x_pruned.pdf'), bbox_inches = 'tight')
property_map.transition_point_predictor[(2,'x')].visualize(max_n_columns=4, skip_constant=True)
plt.savefig(os.path.join(folder_path,'tacrolimus_max_x_pruned.svg'), bbox_inches = 'tight')

sindy_config = {
        'M': dataset.M, 
        't_grid':dataset.get_t_grid(),
        'library':'polynomial',
        'weak':False,
        'sparsity':5,
    }
sindy_bench = bench.SINDYBenchmark(**sindy_config)
results, sindy = bench.run_benchmark_random_splits(dataset, sindy_bench, n_trials=1, n_tune=30, seed=seed, experiment_name="Pharma_real")
mse, std = sindy_bench.evaluate_on_test(sindy)

print("Performance of SINDY on test set: ", mse, std)
save_results('SINDY-5', mse, std)

# Save the results
equations_path = os.path.join(folder_path,'equations.txt')
with open(equations_path, 'a') as f:
    f.write(sindy[0].equations()[0])

sindy_20_config = {
        'M': dataset.M, 
        't_grid':dataset.get_t_grid(),
        'library':'polynomial',
        'weak':False,
        'sparsity':20,
    }
sindy_20_bench = bench.SINDYBenchmark(**sindy_20_config)

results, sindy_20 = bench.run_benchmark_random_splits(dataset, sindy_20_bench, n_trials=1, n_tune=30, seed=seed, experiment_name="Pharma_real")

mse, std = sindy_20_bench.evaluate_on_test(sindy_20)
save_results('SINDY-20', mse, std)

node_config = {
    'max_epochs': 200,
    'batch_size': 32,
    'device': 'cpu',
    'lr': 1e-3,
    'M':dataset.get_M(),
    'K':dataset.get_K(on_hot_encode=True) - len(dataset.x0.keys()),
}

latentodebechmark = bench.LatentODEBenchmark(node_config)
summary, LatentODE = bench.run_benchmark_random_splits(dataset, latentodebechmark, n_trials=1, n_tune=20, seed=seed, experiment_name="Pharma_real")

mse, std = latentodebechmark.evaluate_on_test(LatentODE)
save_results('LatentODE', mse, std)

node_config = {
    'max_epochs': 200,
    'batch_size': 32,
    'device': 'cpu',
    'lr': 1e-3,
    'M':dataset.get_M(),
    'K':dataset.get_K(on_hot_encode=True) - len(dataset.x0.keys()),
    'augmented':False
}

nodebechmark = bench.NeuralODEBenchmark(node_config)
summary, NODE = bench.run_benchmark_random_splits(dataset, nodebechmark, n_trials=1, n_tune=20, seed=seed, experiment_name="Pharma_real")
mse, std = nodebechmark.evaluate_on_test(NODE)
save_results('NODE', mse, std)

node_config = {
    'max_epochs': 200,
    'batch_size': 32,
    'device': 'cpu',
    'lr': 1e-3,
    'M':dataset.get_M(),
    'K':dataset.get_K(on_hot_encode=True) - len(dataset.x0.keys()),
    'augmented':True
}

anodebechmark = bench.NeuralODEBenchmark(node_config)
summary, ANODE = bench.run_benchmark_random_splits(dataset, anodebechmark, n_trials=1, n_tune=20, seed=seed, experiment_name="Pharma_real")
mse, std = anodebechmark.evaluate_on_test(ANODE)
save_results('ANODE', mse, std)

wsindy_config = {
        'M': dataset.M, 
        't_grid':dataset.get_t_grid(),
        'library':'polynomial',
        'weak':True,
        'sparsity':5,
    }
wsindy_bench = bench.SINDYBenchmark(**wsindy_config)
results, wsindy = bench.run_benchmark_random_splits(dataset, wsindy_bench, n_trials=1, n_tune=30, seed=seed, experiment_name="Pharma_real")
mse, std = wsindy_bench.evaluate_on_test(wsindy)
save_results('WSINDY-5', mse, std)

wsindy_20_config = {
        'M': dataset.M, 
        't_grid':dataset.get_t_grid(),
        'library':'polynomial',
        'weak':True,
        'sparsity':20,
    }
wsindy_20_bench = bench.SINDYBenchmark(**wsindy_20_config)
results, wsindy_20 = bench.run_benchmark_random_splits(dataset, wsindy_20_bench, n_trials=1, n_tune=30, seed=seed, experiment_name="Pharma_real")
mse, std = wsindy_20_bench.evaluate_on_test(wsindy_20)
save_results('WSINDY-20', mse, std)

tacrolimusorg = bench.TacrolimusOriginalBenchmark()
results, tac_model = bench.run_benchmark_random_splits(dataset, tacrolimusorg, n_trials=1, n_tune=0, seed=seed, experiment_name="Pharma_real")
mse, std = tacrolimusorg.evaluate_on_test(tac_model)
save_results('Tacrolimus-Original', mse, std)