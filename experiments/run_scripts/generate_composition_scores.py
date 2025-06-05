import sys
sys.path.append('..')
import argparse
from experiments.datasets import *
from experiments.benchmark import *
from experiments.benchmark import _generate_seeds
from episode.api import BSplineBasisFunctions, create_full_composition_library
from experiments.utils import *

if __name__ == "__main__":
    
    # Get the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset")
    parser.add_argument("seed", type=int, default=0, help="Seed")
    parser.add_argument("--trial", action="store_true", help="Run a trial")
    parser.add_argument("--noise_std", type=float, default=0.01)
    args = parser.parse_args()

    # Run the experiment
    if args.trial:
        n_trials = 2
        n_tune = 1
        n_samples = 20
        max_epochs = 10
    else:
        n_trials = 5
        n_tune = 10
        n_samples = 500
        max_epochs = 100
    
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
    
    n_motifs = 8 if args.dataset == 'bike' else 4
    is_infinite = False if args.dataset in ['bike', 'beta'] else True
    full_comp_library = create_full_composition_library(n_motifs,is_infinite=is_infinite, simplified=True)
    composition_libraries = {m: create_full_composition_library(n_motifs,is_infinite=is_infinite, simplified=True) for m in range(M)}
    config = {
        't_range': dataset.t_range,
        'M':M,
        'basis_functions':basis_functions,
        'composition_libraries':composition_libraries,
        'opt_config':opt_config,
        'dt_config':dt_config,
        'x0_dict':dataset.x0,
        'verbose':True,
    }

    model = PSODE(**config, seed=args.seed)

    V, T, Y = dataset.get_V_T_Y()

    time_start = time.time()
    composition_maps_dict = model.fit_composition_maps(V, T, Y)
    time_end = time.time()

    folder = get_composition_scores_folder_path("PSODE-more", dataset.get_name())
    folder_biases = get_composition_scores_folder_path("PSODE-biases_more", dataset.get_name())
    # Check if the folder exists and create it if it doesn't
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder_biases):
        os.makedirs(folder_biases)

    # Save the training time to a file
    with open(os.path.join(folder, 'training_time.txt'), 'w') as f:
        f.write(str(time_end - time_start))

    training_seeds = _generate_seeds(n_trials, args.seed)

    composition_libraries_with_biases = get_inductive_bias_composition_libraries(args.dataset)

    for m in range(M):
        model.single_psodes[m].save_composition_scores_df(folder,f'whole_{model.seed}_{m}')
        df = pd.read_csv(os.path.join(folder, f'composition_scores_whole_{model.seed}_{m}.csv'))

        biases_composition_indices = []
        for bias_composition in composition_libraries_with_biases[m]:
            for i, composition in enumerate(composition_libraries[m]):
                if bias_composition == composition:
                    biases_composition_indices.append(i)
                    break
        assert len(biases_composition_indices) == len(composition_libraries_with_biases[m])

        for training_seed in training_seeds:
            train_indices, val_indices, test_indices = generate_indices(len(dataset), 0.7, 0.15, training_seed)
            train_val_indices = train_indices + val_indices
            subset_df = df.iloc[train_val_indices,:]
            subset_biases_df = df.iloc[train_val_indices,biases_composition_indices]

            file_name = get_composition_scores_file_name(training_seed, m)

            subset_df.to_csv(os.path.join(folder, file_name), index=False)
            subset_biases_df.to_csv(os.path.join(folder_biases, file_name), index=False)
            
    