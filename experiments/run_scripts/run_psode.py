import sys
sys.path.append('..')
import argparse
from experiments.datasets import *
from experiments.benchmark import *
from episode.api import BSplineBasisFunctions, create_full_composition_library
from experiments.utils import *

if __name__ == "__main__":
    
    # Get the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset")
    parser.add_argument("seed", type=int, default=0, help="Seed")
    parser.add_argument('--dtw', action=argparse.BooleanOptionalAction)
    parser.add_argument("--trial", action="store_true", help="Run a trial")
    parser.add_argument("--name", type=str, default="Untitled")
    parser.add_argument("--biases", action="store_true")
    parser.add_argument("--noise_std", type=float, default=0.01)
    args = parser.parse_args()

    # Run the experiment
    if args.trial:
        n_trials = 2
        n_tune = 1
        experiment_name = "Trial"
        n_samples = 20
        max_epochs = 10
    else:
        n_trials = 5
        n_tune = 20
        experiment_name = args.name
        n_samples = 500
        max_epochs = 200
    
    n_measurements = 20
    noise_std = args.noise_std

    dataset_config = {
        'n_samples': n_samples,
        'n_measurements': n_measurements,
        'noise_std': noise_std,
        'seed': args.seed
    }

    if args.dataset == "SIR":
        dataset = get_SIR_dataset(**dataset_config)
    elif args.dataset == "real_pharma":
        dataset = get_real_pharma_dataset()
    elif args.dataset == "pk":
        dataset = get_pk_dataset(**dataset_config)
    elif args.dataset == "bike":
        dataset = get_bike_dataset()
        if args.trial:
            dataset, _ = dataset.split(0.02)
    elif args.dataset == "beta":
        dataset = get_beta_2_dataset(**dataset_config)
    elif args.dataset == "tumor":
        dataset = get_synthetic_tumor_dataset(**dataset_config)
    elif args.dataset == "HIV":
        dataset = get_HIV_dataset(**dataset_config)
    else:
        raise ValueError("Invalid dataset")
    
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
        'dtw':args.dtw
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
    
    if args.biases:
        composition_libraries = get_inductive_bias_composition_libraries(args.dataset)
        subtype = "biases_more"
    else:
        n_motifs = 8 if args.dataset == 'bike' else 4
        is_infinite = False if args.dataset in ['bike', 'beta'] else True
        full_comp_library = create_full_composition_library(n_motifs,is_infinite=is_infinite, simplified=True)
        composition_libraries = {m: create_full_composition_library(n_motifs,is_infinite=is_infinite, simplified=True) for m in range(M)}
        subtype = "more"
    
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
    results, model = run_benchmark_random_splits(dataset, baseline, n_trials=n_trials, n_tune=0, seed=args.seed, experiment_name=experiment_name)
    