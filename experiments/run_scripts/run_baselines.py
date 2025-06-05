import sys
sys.path.append('..')
import argparse
from experiments.datasets import *
from experiments.benchmark import *

if __name__ == "__main__":
    
    # Get the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset")
    parser.add_argument("model", type=str, help="Model")
    parser.add_argument("seed", type=int, default=0, help="Seed")
    parser.add_argument("n_tune", type=int)
    parser.add_argument("--trial", action="store_true", help="Run a trial")
    parser.add_argument("--name", type=str, default="Untitled")
    parser.add_argument("--noise_std", type=float, default=0.01)
    args = parser.parse_args()

    # Run the experiment
    if args.trial:
        n_trials = 1
        n_tune = 1
        experiment_name = "Trial"
        n_samples = 20
        max_epochs = 10
    else:
        n_trials = 5
        n_tune = args.n_tune
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
    elif args.dataset == "beta":
        dataset = get_beta_2_dataset(**dataset_config)
    elif args.dataset == "tumor":
        dataset = get_synthetic_tumor_dataset(**dataset_config)
    elif args.dataset == "HIV":
        dataset = get_HIV_dataset(**dataset_config)
    else:
        raise ValueError("Invalid dataset")
    
    
    node_config = {
            'max_epochs': max_epochs,
            'batch_size': 32,
            'device': 'cpu',
            'lr': 1e-3,
            'M':dataset.get_M(),
            'K':dataset.get_K(on_hot_encode=True) - len([i for i in dataset.x0.values() if isinstance(i,int)]),
        }
    
    sindy_config = {
        'M': dataset.M, 
        't_grid':dataset.get_t_grid(),
        'library':'polynomial'
    }

    if 'SINDy' in args.model:
        if 'W' in args.model:
            sindy_config['weak'] = True
        else:
            sindy_config['weak'] = False
        sparsity = int(args.model.split('-')[-1])
        sindy_config['sparsity'] = sparsity
        baseline = SINDYBenchmark(**sindy_config)
    elif 'NODE' in args.model:
        if 'A' in args.model:
            node_config['augmented'] = True
        else:
            node_config['augmented'] = False
        baseline = NeuralODEBenchmark(node_config)
    elif args.model == "LatentODE":
        baseline = LatentODEBenchmark(node_config)
    
    results, model = run_benchmark_random_splits(dataset, baseline, n_trials=n_trials, n_tune=n_tune, seed=args.seed, experiment_name=experiment_name)
    