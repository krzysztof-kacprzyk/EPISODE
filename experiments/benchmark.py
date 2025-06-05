import optuna
import os
import json
import pickle
import numpy as np
from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import torch
import pandas as pd
import time
import copy
from datetime import datetime
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import pysindy as ps
from experiments.baselines.latent_ode import LatentODERegressor
from experiments.baselines.neural_ode import NeuralODERegressor
from experiments.baselines.tacrolimus import TacrolimusBaseline
from experiments.utils import get_composition_scores_folder_path, get_composition_scores_file_name
from episode.api import PSODE
import torch.nn as nn
import itertools


INF = 1.0e9

def mean_rmse(y_true_list, y_pred_list):
    """
    Compute the mean RMSE between the true and predicted values.

    Parameters:
    y_true_list (list of numpy.ndarray): True values, shape (n_samples, n_timepoints, n_dimensions)
    y_pred_list (list of numpy.ndarray): Predicted values, shape (n_samples, n_timepoints, n_dimensions)
    Returns:
    float: Mean RMSE.
    """
    
    diff_list = [y_true - y_pred for y_true, y_pred in zip(y_true_list, y_pred_list)]
    M = y_true_list[0].shape[1]
    norm_squared_list = [np.sum(diff**2,axis=1) / M for diff in diff_list]
    rmse_per_sample = [np.sqrt(np.mean(norm_squared)) for norm_squared in norm_squared_list]
    return np.mean(rmse_per_sample)

def std_rmse(y_true_list, y_pred_list):
    """
    Compute the standard deviation of RMSE between the true and predicted values.
    
    Parameters:
    y_true_list (list of numpy.ndarray): True values, shape (n_samples, n_timepoints, n_dimensions)
    y_pred_list (list of numpy.ndarray): Predicted values, shape (n_samples, n_timepoints, n_dimensions)
    Returns:
    float: Standard deviation of RMSE.
    """

    diff_list = [y_true - y_pred for y_true, y_pred in zip(y_true_list, y_pred_list)]
    M = y_true_list[0].shape[1]
    norm_squared_list = [np.sum(diff**2,axis=1) / M for diff in diff_list]
    rmse_per_sample = [np.sqrt(np.mean(norm_squared)) for norm_squared in norm_squared_list]
    return np.std(rmse_per_sample)




def interpolate_nans(y):
    """
    Interpolate np.nan values in a one-dimensional numpy array using linear interpolation.
    If interpolation is impossible, return an array of zeros.

    Parameters:
    y (numpy.ndarray): Input one-dimensional array with possible np.nan values.

    Returns:
    numpy.ndarray: Array with np.nan values imputed or zeros if interpolation is impossible.
    """
    threshold = 10
    y = np.where(np.abs(y) > threshold, np.nan, y)
    y = np.where(np.abs(y) < -threshold, np.nan, y)
    y = np.asarray(y, dtype=np.float64)
    x = np.arange(len(y))
    mask = np.isnan(y)
    valid = ~mask

    # Check if interpolation is possible
    if np.count_nonzero(valid) < 2:
        if np.count_nonzero(valid) == 1:
            y[mask] = y[valid][0]
            return y
        else:
            return np.zeros_like(y)

    # Perform linear interpolation
    f = interp1d(
        x[valid],
        y[valid],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate')
    
    y[mask] = f(x[mask])
    return y



def clip_to_finite(y):
    # Fill np.nan values with INF
    y = np.where(np.isnan(y), INF, y)
    return np.clip(y,-INF,INF)

def _generate_seeds(n_trials, seed):
    """Generate seeds for training."""
    rng = np.random.default_rng(seed)
    training_seeds = rng.integers(0, 2**31 - 1, size=1000)
    training_seeds = [s.item() for s in training_seeds[:n_trials]]
    return training_seeds

def generate_indices(n, train_size, val_size, seed=0):
    gen = np.random.default_rng(seed)
    train_indices = gen.choice(n, int(n*train_size), replace=False)
    train_indices = [i.item() for i in train_indices]
    val_indices = gen.choice(list(set(range(n)) - set(train_indices)), int(n*val_size), replace=False)
    val_indices = [i.item() for i in val_indices]
    test_indices = list(set(range(n)) - set(train_indices) - set(val_indices))
    return train_indices, val_indices, test_indices

def run_benchmark_random_splits(dataset, method, dataset_split = [0.7,0.15,0.15], n_trials=10, n_tune=100, seed=0, benchmarks_dir='results', experiment_name='untitled'):
    """
    Runs a set of benchmarks on a dataset
    Args:
    """

    # Add a row to the DataFrame
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    experiment_dir = os.path.join(benchmarks_dir, experiment_name)

    # Check if there exists a file summary.json in the benchmarks directory
    if os.path.exists(os.path.join(experiment_dir, 'summary.csv')):
        # Load as a DataFrame
        summary = pd.read_csv(os.path.join(experiment_dir, 'summary.csv'))
    else:
        # Create folder if does not exist
        os.makedirs(experiment_dir, exist_ok=True)
        # Create
        summary = pd.DataFrame(columns=['timestamp', 'dataset', 'method', 'n_trials', 'n_tune', 'train_size', 'val_size', 'seed', 'test_loss_mean', 'test_loss_std', 'time_elapsed'])
        # Save the summary
        summary.to_csv(os.path.join(experiment_dir, 'summary.csv'), index=False)
        

   
    time_start = time.time()
    benchmark = method
    losses, single_run_time, model = benchmark.run_random_splits(dataset, dataset_split, 
                                                                 n_trials=n_trials, 
                                                                 n_tune=n_tune, 
                                                                 seed=seed, 
                                                                 experiment_dir=experiment_dir, 
                                                                 timestamp=timestamp,
                                                                 benchmarks_dir=benchmarks_dir)
    time_end = time.time()

    # Add a new row to the summary
    results = {
        'timestamp': [timestamp],
        'dataset_name': [dataset.get_name()],
        'method': [method.get_name()],
        'n_trials': [n_trials],
        'n_tune': [n_tune],
        'train_size': [dataset_split[0]],
        'val_size': [dataset_split[1]],
        'seed': [seed],
        'test_loss_mean': [np.mean(losses)],
        'test_loss_std': [np.std(losses)],
        'time_elapsed': time_end - time_start,
        'single_run_time': single_run_time
        }

    # concatenate the results to the summary
    summary = pd.concat([summary, pd.DataFrame(results)], ignore_index=True)
    
    # Save the summary
    summary.to_csv(os.path.join(experiment_dir, 'summary.csv'), index=False)

    return summary, model


class BaseBenchmark(ABC):
    """Base class for benchmarks."""

    def __init__(self):
        self.name = self.get_name()


    def tune(self, n_trials, seed, experiment_dir):
        """Tune the benchmark."""

        def objective(trial):
            model = self.get_model_for_tuning(trial, seed)
            model = self.train(model, tuning=True)
            val_loss, _ = self.evaluate_on_val(model)
            print(f'[Trial {trial.number}] val_loss: {val_loss}')
            return val_loss
        
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler,direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        best_hyperparameters = best_trial.params

        print('[Best hyperparameter configuration]:')
        print(best_hyperparameters)

        tuning_dir = os.path.join(experiment_dir, self.name, self.timestamp, 'tuning')
        os.makedirs(tuning_dir, exist_ok=True)

        # Save best hyperparameters
        hyperparam_save_path = os.path.join(tuning_dir, f'hyperparameters.json')
        with open(hyperparam_save_path, 'w') as f:
            json.dump(best_hyperparameters, f)
        
        # Save optuna study
        study_save_path = os.path.join(tuning_dir, f'study_{seed}.pkl')
        with open(study_save_path, 'wb') as f:
            pickle.dump(study, f)

        # Save trials dataframe
        df = study.trials_dataframe()
        df.set_index('number', inplace=True)
        df_save_path = os.path.join(tuning_dir, f'trials_dataframe.csv')
        df.to_csv(df_save_path)

        print(f'[Tuning complete], saved tuning results to {tuning_dir}')

        return best_hyperparameters
    
    def run_random_splits(self, dataset, dataset_split, n_trials, n_tune, seed, experiment_dir, timestamp, benchmarks_dir, **kwargs):
        """Run the benchmark."""
        self.experiment_dir = experiment_dir
        self.timestamp = timestamp
        self.benchmarks_dir = benchmarks_dir

        results_folder = os.path.join(experiment_dir, self.name, self.timestamp, 'final')
        self.results_folder = results_folder

        # Create a numpy random generator
        rng = np.random.default_rng(seed)

        # Generate seeds for training
        training_seeds = _generate_seeds(n_trials, seed)

        print(f"[Testing for {n_trials} trials]")

        # Train the model n_trials times
        test_losses = []
        run_times = []
        for i in range(n_trials):
            if n_trials == 1:
                training_seed = seed
            else:
                training_seed = training_seeds[i]

            print(f"[Training trial {i+1}/{n_trials}] seed: {training_seed}")

            # Generate train, validation, and test indices for tuning
            train_indices, val_indices, test_indices = generate_indices(len(dataset), dataset_split[0], dataset_split[1], seed=training_seed)
        
            # Prepare the data
            self.prepare_data(dataset, train_indices, val_indices, test_indices)

            # Tune the model
            if n_tune > 0:
                print(f"[Tuning for {n_tune} trials]")
                best_hyperparameters = self.tune(n_trials=n_tune, seed=training_seed, experiment_dir=experiment_dir)
            else:
                print(f"[No tuning, using default hyperparameters]")
                best_hyperparameters = None

            model = self.get_final_model(best_hyperparameters, training_seed)
            
            start_time = time.time()
            model = self.train(model)
            test_loss, _ = self.evaluate_on_test(model)
            end_time = time.time()
            print(f"[Test loss]: {test_loss}")
            test_losses.append(test_loss)
            run_times.append(end_time - start_time)

            # Save the losses
            df = pd.DataFrame({'seed':training_seeds[:i+1],'test_loss': test_losses, 'run_time': run_times})
            os.makedirs(results_folder, exist_ok=True)
            test_losses_save_path = os.path.join(results_folder, f'results.csv')
            df.to_csv(test_losses_save_path, index=False)

        average_single_run_time = np.mean(run_times)
        return test_losses, average_single_run_time, model

    def prepare_data(self, dataset, train_indices, val_indices, test_indices):
        """Prepare the data for the benchmark."""

        self.dataset = dataset

        self.dataset_train = dataset._create_subset(train_indices, suffix='train')
        self.dataset_val = dataset._create_subset(val_indices, suffix='val')
        self.dataset_test = dataset._create_subset(test_indices, suffix='test')

        train_val_indices = train_indices + val_indices
        self.dataset_train_val = dataset._create_subset(train_val_indices, suffix='train_val')

        self.fit_one_hot_encoder()

    def evaluate_on_val(self, model):
        Y_val = self.dataset_val.get_Y()
        Y_pred = self.predict(model, self.dataset_val)
        val_loss = mean_rmse(Y_val, Y_pred)
        val_std = std_rmse(Y_val, Y_pred)
        return val_loss, val_std
    
    def evaluate_on_test(self, model):
        Y_test = self.dataset_test.get_Y()
        Y_pred = self.predict(model, self.dataset_test)
        test_loss = mean_rmse(Y_test, Y_pred)
        test_std = std_rmse(Y_test, Y_pred)
        return test_loss, test_std
    
    def split_V_and_X0(self, V, x0_dict, M):
        """
        Args:
        V (pd.DataFrame): Input data of shape (n_samples, n_features)
        x0_dict (dict): Dictionary of initial conditions

        Returns:
        X0 (np.ndarray): Initial conditions of shape (n_samples, M)
        no_x0_V (pd.DataFrame): Input data with initial conditions removed
        """
        X0 = np.zeros((V.shape[0],M))
        for i in range(M):
            if i in x0_dict.keys():
                if isinstance(x0_dict[i], int):
                    X0[:,i] = V.iloc[:,x0_dict[i]].to_numpy()
                elif isinstance(x0_dict[i], float):
                    X0[:,i] = x0_dict[i]
            else:
                X0[:,i] = 0
        indices_to_drop = [i for i in x0_dict.values() if isinstance(i, int)]
        no_x0_V = V.drop(columns=[V.columns[i] for i in indices_to_drop])
        return X0, no_x0_V
    
    def convert_V_to_numpy(self, V, categorical_variables):

        return self.transform_one_hot(V)
    
    def fit_one_hot_encoder(self):
        V = self.dataset.get_V().copy()
        initial_conditions_cols = [self.dataset.V.columns[i] for i in self.dataset.x0.values() if isinstance(i, int)]
        categorical_columns = V.columns[self.dataset.categorical_features_indices]
        column_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_columns)
            ],
            remainder='passthrough'  # keeps the other columns (numerical) as they are
        )
        self.one_hot_encoder = column_transformer
        V = V.drop(columns=initial_conditions_cols)
        self.one_hot_encoder.fit(V)
    
    def transform_one_hot(self, V):
        if not hasattr(self, 'one_hot_encoder'):
            raise ValueError("One-hot encoder not fitted")
        return self.one_hot_encoder.transform(V)
        


    @abstractmethod
    def train(self, model, tuning=False):
        """
        Train the benchmark. Returns a dictionary with train, validation, and test loss
        Returns:
            model: trained model
        """
        pass

    @abstractmethod
    def predict(self, model, dataset):
        """
        Predict the values.
        
        Args:
        model (object): Model object.
        dataset (object): Dataset object.

        Returns:
        numpy.ndarray: Predicted values of shape (n_samples, n_timepoints, n_dimensions) or a list of n_sample numpy arrays of shape (n_timepoints_d, n_dimensions)
        """
        pass
       
    @abstractmethod
    def get_model_for_tuning(self, trial, seed):
        """Get the model."""
        pass

    @abstractmethod
    def get_final_model(self, hyperparameters, seed):
        """Get the model."""
        pass
    
    @abstractmethod
    def get_name(self):
        """Get the name of the benchmark."""
        pass


class PSODEBenchmark(BaseBenchmark):
    """PSODE"""

    def __init__(self, config, benchmarks_dir='results'):
        self.config = config
        self.benchmarks_dir = benchmarks_dir
        super().__init__()

    def get_name(self):
        return 'PSODE-'+self.config['subtype']
    
    def get_model_for_tuning(self, trial, seed):
        """Get model for tuning."""

        # There are no hyperparameters to tune at this stage
        # PSODE does its own hyperparameter tuning
        config = copy.deepcopy(self.config)
        del config['subtype']
        model = PSODE(**config, seed=seed)
        return model
       
    def get_final_model(self, parameters, seed):
        """Get model for testing."""

        # There are no hyperparameters to tune at this stage
        # PSODE does its own hyperparameter tuning
        config = copy.deepcopy(self.config)
        del config['subtype']
        model = PSODE(**config, seed=seed)
        return model

    def _get_composition_scores_path(self, dataset_name, seed, m):
        folder_path = get_composition_scores_folder_path(self.name, dataset_name, benchmarks_dir=self.benchmarks_dir)
        file_name = get_composition_scores_file_name(seed, m)
        return os.path.join(folder_path, file_name)

    def _check_if_composition_scores_exist(self, dataset_name, seed, m):
        filepath = self._get_composition_scores_path(dataset_name, seed, m)
        return os.path.exists(filepath)
    
    def _load_composition_scores(self, dataset_name, seed, m):
        filepath = self._get_composition_scores_path(dataset_name, seed, m)
        composition_scores = pd.read_csv(filepath)
        return composition_scores.to_numpy()
        
    def train(self, model, tuning=False):

        # We combine the training and validation data because we do not do hyperparameter tuning
        # We will use validation data for early stopping and validation throughout

        V_train_val, T_train_val, Y_train_val = self.dataset_train_val.get_V_T_Y(return_numpy=False, m=None)

        composition_maps_dict = model.composition_maps # Normally that would be empty
        composition_scores_dict = {}

        for m in range(self.config['M']):
            if m not in composition_maps_dict.keys():
                # Try to load the composition scores
                # Check if the composition scores exist
                if self._check_if_composition_scores_exist(self.dataset.get_name(), model.seed, m):
                    print(f'Composition scores for m={m} exist')
                    composition_scores = self._load_composition_scores(self.dataset.get_name(), model.seed, m)
                    composition_scores_dict[m] = composition_scores
                else:
                    print(f'Composition scores for m={m} do not exist')
        
        print('Fitting composition maps...')
        new_composition_maps_dict = model.fit_composition_maps(V_train_val, T_train_val, Y_train_val, composition_scores_dict=composition_scores_dict)
        print('Composition maps fitted.')
        # Only update the composition maps that were not already present. 
        # This means some coposition maps are fitted unnecessarily
        # but it's not a huge computational cost
        for m in range(self.config['M']):
            if m not in composition_maps_dict.keys():
                composition_maps_dict[m] = new_composition_maps_dict[m]

        model.fit(V_train_val, T_train_val, Y_train_val, composition_maps=composition_maps_dict)

        for m in range(self.config['M']):
            model.single_psodes[m].save_composition_scores_df(self.results_folder,f'{model.seed}_{m}')
        
        return model
    
    def predict(self, model, dataset):
        V, T, Y = dataset.get_V_T_Y(return_numpy=False, m=None)
        return model.predict(V, T)
    

class TacrolimusOriginalBenchmark(BaseBenchmark):
    """Tacrolimus original benchmark"""

    def __init__(self):
        super().__init__()
    
    def get_name(self):
        return 'TacrolimusOriginal'
    
    def get_model_for_tuning(self, trial, seed):
        model = TacrolimusBaseline()
        return model
    
    def get_final_model(self, hyperparameters, seed):
        model = TacrolimusBaseline()
        return model
    
    def train(self, model, tuning=False):
        return model
    
    def predict(self, model, dataset):
        V, T, Y = self.dataset_val.get_V_T_Y(return_numpy=False, m=None)
        return model.predict(V, T)
    
class NeuralODEBenchmark(BaseBenchmark):
    
    def __init__(self, config):
        self.config = config
        super().__init__()

    def get_name(self):
        if self.config['augmented']:
            return 'ANODE'
        else:
            return 'NeuralODE'
    
    def get_model_for_tuning(self, trial, seed):

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        layer_sizes = []
        for i in range(num_layers):
            units = trial.suggest_int(f'layer_{i}_units', 16, 128, log=True)
            layer_sizes.append(units)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        activation_name = trial.suggest_categorical('activation', ['ELU', 'Sigmoid'])
        activation = getattr(nn, activation_name)
        if self.config['augmented']:
            augment_dim = trial.suggest_int('augment_dim', 1, 10)
        else:
            augment_dim = 0

        # Create the model with suggested hyperparameters
        model = NeuralODERegressor(
            M=self.config['M'],
            K=self.config['K'],
            layer_sizes=layer_sizes,
            activation=activation,
            init_method=nn.init.kaiming_normal_ if activation_name == 'ReLU' else nn.init.xavier_normal_,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            solver='rk4',
            # solver_options={'step_size': 0.01},
            device=self.config['device'],
            seed=seed,
            augment_dim=augment_dim
        )

        return model
    
    def get_final_model(self, parameters, seed):
        """Get model for testing."""
    
        if parameters is None:
            # Return a default model
            if self.config['augmented']:
                augment_dim = 5
            else:
                augment_dim = 0
            return NeuralODERegressor(
                M=self.config['M'],
                K=self.config['K'],
                layer_sizes=[64, 64],
                activation=nn.ELU,
                init_method=nn.init.kaiming_normal_,
                learning_rate=self.config['lr'],
                weight_decay=1e-6,
                dropout_rate=0.1,
                solver='rk4',
                # solver='dopri5',
                # solver_options={'step_size': 0.01},
                device=self.config['device'],
                seed=seed,
                augment_dim=augment_dim
            )
        else:    
            # Get the layer sizes
            num_layers = parameters['num_layers']
            layer_sizes = []
            for i in range(num_layers):
                units = parameters[f'layer_{i}_units']
                layer_sizes.append(units)
            activation_name = parameters['activation']
            activation = getattr(nn, activation_name)
            if self.config['augmented']:
                augment_dim = parameters['augment_dim']
            else:
                augment_dim = 0
            return NeuralODERegressor(
                M=self.config['M'],
                K=self.config['K'],
                layer_sizes=layer_sizes,
                activation=activation,
                init_method=nn.init.kaiming_normal_ if parameters['activation'] == nn.ReLU else nn.init.xavier_normal_,
                learning_rate=parameters['learning_rate'],
                weight_decay=parameters['weight_decay'],
                dropout_rate=parameters['dropout_rate'],
                solver='rk4',
                # solver_options={'step_size': 0.1},
                device=self.config['device'],
                seed=seed,
                augment_dim=augment_dim
            )
        
    def train(self, model, tuning=False):
        V_train, T_train, Y_train = self.dataset_train.get_V_T_Y(m=None)
        V_val, T_val, Y_val = self.dataset_val.get_V_T_Y(m=None)
        t_shared = T_train[0]

        X0_train, no_x0_V_train = self.split_V_and_X0(V_train, self.dataset_train.x0, self.config['M'])
        X0_val, no_x0_V_val = self.split_V_and_X0(V_val, self.dataset_val.x0, self.config['M'])
        no_x0_V_train = self.convert_V_to_numpy(no_x0_V_train, self.dataset_train.categorical_features_indices)
        no_x0_V_val = self.convert_V_to_numpy(no_x0_V_val, self.dataset_val.categorical_features_indices)

        model.fit(X0_train, t_shared, Y_train, no_x0_V_train, X0_val, t_shared, Y_val, no_x0_V_val, batch_size=self.config['batch_size'], max_epochs=self.config['max_epochs'], tuning=tuning)
        return model
    
    def predict(self, model, dataset):
        V, T, Y = dataset.get_V_T_Y(m=None)
        t_shared = T[0]

        X0, no_x0_V = self.split_V_and_X0(V, dataset.x0, self.config['M'])

        no_x0_V = self.convert_V_to_numpy(no_x0_V, self.dataset_train.categorical_features_indices)
       
        return model.predict(X0, t_shared, no_x0_V)
    

class LatentODEBenchmark(BaseBenchmark):
    """Latent ODE benchmark"""

    def __init__(self, config):
        self.config = config
        super().__init__()
    
    def get_name(self):
        return 'LatentODE'
    
    def get_model_for_tuning(self, trial, seed):
        
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        encoder_num_layers = trial.suggest_int('encoder_num_layers', 1, 3)
        encoder_layer_sizes = []
        for i in range(encoder_num_layers):
            units = trial.suggest_int(f'encoder_layer_{i}_units', 16, 128, log=True)
            encoder_layer_sizes.append(units)
        decoder_num_layers = trial.suggest_int('decoder_num_layers', 1, 3)
        decoder_layer_sizes = []
        for i in range(decoder_num_layers):
            units = trial.suggest_int(f'decoder_layer_{i}_units', 16, 128, log=True)
            decoder_layer_sizes.append(units)
        odefun_num_layers = trial.suggest_int('odefun_num_layers', 1, 3)
        odefun_layer_sizes = []
        for i in range(odefun_num_layers):
            units = trial.suggest_int(f'odefun_layer_{i}_units', 16, 128, log=True)
            odefun_layer_sizes.append(units)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        activation_name = trial.suggest_categorical('activation', ['ELU', 'Sigmoid'])
        activation = getattr(nn, activation_name)
        latent_dim = trial.suggest_int('latent_dim', 1, 10)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

        # Create the model with suggested hyperparameters
        model = LatentODERegressor(
            M=self.config['M'],
            K=self.config['K'],
            latent_dim=latent_dim,
            encoder_sizes = encoder_layer_sizes,
            decoder_sizes = decoder_layer_sizes,
            odefunc_sizes = odefun_layer_sizes,
            activation=activation,
            init_method=nn.init.xavier_normal_,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            solver='rk4',
            device=self.config['device'],
            seed=seed
        )

        return model
    
    def get_final_model(self, parameters, seed):
        """Get model for testing."""
    
        if parameters is None:
            # Return a default model
            return LatentODERegressor(
                M=self.config['M'],
                K=self.config['K'],
                latent_dim=5,
                encoder_sizes=[64, 64],
                decoder_sizes=[64, 64],
                odefunc_sizes=[64, 64],
                activation=nn.ELU,
                init_method=nn.init.xavier_normal_,
                learning_rate=self.config['lr'],
                weight_decay=1e-6,
                dropout_rate=0.1,
                solver='rk4',
                device=self.config['device'],
                seed=seed
            )
        else:    
            # Get the layer sizes
            encoder_num_layers = parameters['encoder_num_layers']
            encoder_layer_sizes = []
            for i in range(encoder_num_layers):
                units = parameters[f'encoder_layer_{i}_units']
                encoder_layer_sizes.append(units)
            decoder_num_layers = parameters['decoder_num_layers']
            decoder_layer_sizes = []
            for i in range(decoder_num_layers):
                units = parameters[f'decoder_layer_{i}_units']
                decoder_layer_sizes.append(units)
            odefun_num_layers = parameters['odefun_num_layers']
            odefun_layer_sizes = []
            for i in range(odefun_num_layers):
                units = parameters[f'odefun_layer_{i}_units']
                odefun_layer_sizes.append(units)
            activation_name = parameters['activation']
            activation = getattr(nn, activation_name)
            latent_dim = parameters['latent_dim']
            return LatentODERegressor(
                M=self.config['M'],
                K=self.config['K'],
                latent_dim=latent_dim,
                encoder_sizes = encoder_layer_sizes,
                decoder_sizes = decoder_layer_sizes,
                odefunc_sizes = odefun_layer_sizes,
                activation=activation,
                init_method=nn.init.xavier_normal_,
                learning_rate=parameters['learning_rate'],
                weight_decay=parameters['weight_decay'],
                dropout_rate=parameters['dropout_rate'],
                solver='rk4',
                device=self.config['device'],
                seed=seed
            )
        
    def train(self, model, tuning=False):
        V_train, T_train, Y_train = self.dataset_train.get_V_T_Y(m=None)
        V_val, T_val, Y_val = self.dataset_val.get_V_T_Y(m=None)
        t_shared = T_train[0]

        X0_train, no_x0_V_train = self.split_V_and_X0(V_train, self.dataset_train.x0, self.config['M'])
        X0_val, no_x0_V_val = self.split_V_and_X0(V_val, self.dataset_val.x0, self.config['M'])

        no_x0_V_train = self.convert_V_to_numpy(no_x0_V_train, self.dataset_train.categorical_features_indices)
        no_x0_V_val = self.convert_V_to_numpy(no_x0_V_val, self.dataset_val.categorical_features_indices)
       
        model.fit(X0_train, t_shared, Y_train, no_x0_V_train, X0_val, t_shared, Y_val, no_x0_V_val, batch_size=self.config['batch_size'], max_epochs=self.config['max_epochs'], tuning=tuning)
        return model
    
    def predict(self, model, dataset):
        V, T, Y = dataset.get_V_T_Y(m=None)
        t_shared = T[0]

        X0, no_x0_V = self.split_V_and_X0(V, dataset.x0, self.config['M'])

        no_x0_V = self.convert_V_to_numpy(no_x0_V, self.dataset_train.categorical_features_indices)
       
        return model.predict(X0, t_shared, no_x0_V)
        
class SINDYBenchmark(BaseBenchmark):
    """SINDY benchmark"""

    def __init__(self, M, sparsity, weak=False, t_grid=None, library='general'):
        self.M = M
        self.sparsity = sparsity
        self.weak = weak
        self.t_grid = t_grid
        self.library = library
        super().__init__()

    def get_name(self):
        if self.weak:
            return 'WSINDy-'+str(self.sparsity)
        else:
            return 'SINDy-'+str(self.sparsity)

    
    def get_library(self, library_type='general'):

        if self.weak:

            if library_type == 'general':
                library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x*y, lambda x: np.exp(x)]
                library_functions += [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.sin(2*x), lambda x: np.cos(2*x), lambda x: np.sin(3*x), lambda x: np.cos(3*x)]
                
                library_function_names = [lambda x: f"{x}", lambda x: f"{x}^2", lambda x,y: f"{x}{y}", lambda x: f"exp({x})"]
                library_function_names += [lambda x: f"sin({x})", lambda x: f"cos({x})", lambda x: f"sin(2{x})", lambda x: f"cos(2{x})", lambda x: f"sin(3{x})", lambda x: f"cos(3{x})"]
            elif library_type == 'polynomial':
                library_functions = [lambda x: x, lambda x: x * x, lambda x, y: x*y]
                library_function_names = [lambda x: f"{x}", lambda x: f"{x}^2", lambda x,y: f"{x}{y}"]

            library = ps.WeakPDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            spatiotemporal_grid=self.t_grid,
            is_uniform=True,
            K=200,
            include_bias=True
            )
        else:
            polynomial_library = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
            fourier_full_library = ps.FourierLibrary(n_frequencies=3,include_cos=True,include_sin=True)
            exp_only_library = ps.CustomLibrary(library_functions=[lambda x : np.exp(x)], function_names=[lambda x: f"exp({x})"])
            if library_type == 'general':
                library = ps.GeneralizedLibrary([polynomial_library, fourier_full_library, exp_only_library])
            elif library_type == 'polynomial':
                library = polynomial_library

        return library
    
    def get_model_for_tuning(self, trial, seed):

        if self.library == 'tune':
            library_type = trial.suggest_categorical('library_type', ['general', 'polynomial'])
        else:
            library_type = self.library
        
        library = self.get_library(library_type=library_type)
        
        alpha = trial.suggest_float('alpha_miosr', 1e-3, 1, log=True)

        if self.sparsity == 0:
            sparsity = trial.suggest_int('sparsity', 1, 20)
        else:
            sparsity = self.sparsity
        group_sparsity = tuple([sparsity]*self.M)

        optimizer = ps.MIOSR(alpha=alpha,target_sparsity=None,group_sparsity=group_sparsity)

        differentiation_kind = trial.suggest_categorical('differentiation_kind', ['finite_difference', 'spline', 'trend_filtered'])
        if differentiation_kind == 'finite_difference':
            k = trial.suggest_int('k', 1, 5)
            differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
        elif differentiation_kind == 'spline':
            s = trial.suggest_float('s', 1e-3, 1, log=True)
            differentiation_method = ps.SINDyDerivative(kind='spline', s=s)
        elif differentiation_kind == 'trend_filtered':
            order = trial.suggest_int('order', 0, 2)
            alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)
            differentiation_method = ps.SINDyDerivative(kind='trend_filtered', order=order, alpha=alpha)
        elif differentiation_kind == 'smoothed_finite_difference':
            window_length = trial.suggest_int('window_length', 1, 5)
            differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={"window_length":window_length})

        model = ps.SINDy(optimizer=optimizer, differentiation_method=differentiation_method, feature_library=library)

        return (model, seed)
       
    def get_final_model(self, parameters, seed):

        if parameters is None:
            if self.sparsity == 0:
                sparsity = 5
            else:
                sparsity = self.sparsity
            if self.library == 'tune':
                library_type = 'general'
            else:
                library_type = self.library
            alpha = 0.1
            k = 2
            differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
        else:
            if self.library == 'tune':
                library_type = parameters['library_type']
            else:
                library_type = self.library

            if self.sparsity == 0:
                sparsity = parameters['sparsity']
            else:
                sparsity = self.sparsity
            alpha = parameters['alpha_miosr']
            differentiation_kind = parameters['differentiation_kind']
            if differentiation_kind == 'finite_difference':
                k = parameters['k']
                differentiation_method = ps.SINDyDerivative(kind='finite_difference', k=k)
            elif differentiation_kind == 'spline':
                s = parameters['s']
                differentiation_method = ps.SINDyDerivative(kind='spline', s=s)
            elif differentiation_kind == 'trend_filtered':
                order = parameters['order']
                alpha = parameters['alpha']
                differentiation_method = ps.SINDyDerivative(kind='trend_filtered', order=order, alpha=alpha)
            elif differentiation_kind == 'smoothed_finite_difference':
                window_length = parameters['window_length']
                differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={"window_length":window_length})
        group_sparsity = tuple([sparsity]*self.M)
        optimizer = ps.MIOSR(alpha=alpha,target_sparsity=None,group_sparsity=group_sparsity)
        library = self.get_library(library_type=library_type)
        model = ps.SINDy(optimizer=optimizer, differentiation_method=differentiation_method, feature_library=library)

        return (model, seed)
    
    # def simulate(model, x0, t, u):

    def get_U(self,V,T):
        n_samples = V.shape[0]
        U = []
        for i in range(n_samples):
            n_measurements = T[i].shape[0]
            row = V[[i],:] # 1 x K
            U_d = np.tile(row, (n_measurements,1)) # N_d x K
            U_d = np.concatenate((U_d, T[i].reshape(-1,1)), axis=1) # N_d x K+1
            U.append(U_d)
        return U

    def train(self, model, tuning=False):

        model, seed = model

        V_train, T_train, Y_train = self.dataset_train.get_V_T_Y(m=None)

        X0_train, no_x0_V_train = self.split_V_and_X0(V_train, self.dataset_train.x0, self.dataset_train.M)

        no_x0_V_train = self.convert_V_to_numpy(no_x0_V_train, self.dataset_train.categorical_features_indices)

        U = self.get_U(no_x0_V_train, T_train)

        np.random.seed(seed)

        model.fit(x=Y_train, t=T_train, u=U, multiple_trajectories=True)

        return (model, seed)
    
    
    def predict(self, model, dataset):

        model, seed = model
        
        V, T, Y = dataset.get_V_T_Y(m=None)

        X0, no_x0_V = self.split_V_and_X0(V, dataset.x0, dataset.M)

        no_x0_V = self.convert_V_to_numpy(no_x0_V, dataset.categorical_features_indices)

        if self.weak:
            library_functions = model.feature_library.functions
            M = self.M
            n_all_x = M + no_x0_V.shape[1] + 1 # 1 for time
            actual_functions = []
            for i in range(len(library_functions)):
                if library_functions[i].__code__.co_argcount == 1:
                    for j in range(n_all_x):
                        actual_functions.append((lambda val, val2: lambda all_x: library_functions[val](all_x[val2]))(i,j))
                elif library_functions[i].__code__.co_argcount == 2:
                    combs = list(itertools.combinations(range(n_all_x),2))
                    for comb in combs:
                        actual_functions.append((lambda val, val2, val3: lambda all_x: library_functions[val](all_x[val2],all_x[val3]))(i,comb[0],comb[1]))

        def get_control_function(v):
            """
            Args:
                v (np.ndarray): Input data of shape (n_features)
            """
            def u(t):
                return np.concatenate([v, np.array([t])]) 
            return u
        
        def simulate(model, x0, t, u):
            functions = actual_functions
            coefficients = model.coefficients()
            def derivative(x, t):
                """
                Args:
                    x (np.ndarray): Input data of shape (M)
                    t (float): Time point
                """
                u_val = u(t)
                xu = np.concatenate([x, u_val], axis=0)
                res = []
                for m in range(M):
                    res_m = 0
                    for i, f in enumerate(functions):
                        res_m += coefficients[m][i+1] * f(xu)
                    res_m +=coefficients[m][0]
                    res.append(res_m)
                return np.array(res)
            y, infodict = odeint(derivative, x0, t=t, full_output=True)
            y = np.nan_to_num(y)
            return y
        

        Y_pred = []
        for x0, v, t in zip(X0, no_x0_V, T): # x0: M, v: K, t: N_d
            u = get_control_function(v)
            try:
                if self.weak:
                    y_pred = np.clip(simulate(model, x0=x0,t=t,u=u),-INF,INF)
                else:
                    y_pred = np.clip(model.simulate(x0=x0,t=t,u=u,integrator='odeint'),-INF,INF)
            except Exception as e:
                print(e)
                y_pred = np.zeros((t.shape[0],dataset.M))
            Y_pred.append(y_pred)

        return Y_pred