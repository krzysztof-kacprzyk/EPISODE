import copy
from datetime import datetime
import itertools
import os
from scipy.interpolate import BSpline
import numpy as np
from abc import ABC, abstractmethod
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import optuna
import logging
from episode.lit_module import LitSketchODE
import matplotlib.pyplot as plt

import multiprocessing
from episode.model_numpy import calculate_loss
import pandas as pd
from tqdm import tqdm
import episode.utils as utils
import seaborn as sns
from episode.decision_tree import DecisionTreeClassifier
from episode.gam import GAM, CustomPropertyFunction, ShapeFunction, ZeroPropertyFunction
from episode.reconstruct_cubic import ApproximatePredictiveModel, PredictiveModel
from episode.property_map import PropertyMapExtractor

INF = 1e9

# def process_sample(info):
#         sample, compositions, t_range, seed, sample_id = info
#         x0, t, y = sample
#         sample_scores = {'id': sample_id}
#         for i, composition in enumerate(compositions):
#             loss, model = calculate_loss(composition, t_range, x0, t, y, seed=seed, evaluate_on_all_data=True)
#             if np.isnan(loss):
#                 loss = INF
#             sample_scores[i] = loss
#         return sample_scores

def process_sample_one_comp(info):
    sample, composition_id, composition, t_range, seed, sample_id, train_on_all_data = info
    x0, t, y = sample
    sample_scores = {'id': sample_id}
    if utils.is_unbounded_composition(composition):
        train_on_all_data = train_on_all_data
    else:
        train_on_all_data = True
    loss, model = calculate_loss(composition, t_range, x0, t, y, seed=seed, train_on_all_data=train_on_all_data, evaluate_on_all_data=True)
    if np.isnan(loss):
        loss = INF
    sample_scores['col'] = composition_id
    sample_scores['val'] = loss
    return sample_scores

def combine_dicts_to_dataframe(dicts):
    """
    Combines a list of dictionaries into a single pandas DataFrame.

    Args:
        dicts (list of dict): List of dictionaries in the form {'id': id, 'col': col, 'val': val}.

    Returns:
        pd.DataFrame: A DataFrame with columns as specified by 'col',
                      values as specified by 'val', and rows aligned by 'id'.
    """
    # Convert each dictionary into a smaller DataFrame
    dataframes = []
    for d in dicts:
        temp_df = pd.DataFrame({
            'id': [d['id']],
            d['col']: [d['val']]
        })
        dataframes.append(temp_df)

    # Merge all DataFrames on 'id'
    result_df = pd.concat(dataframes, ignore_index=True).pivot_table(
        index='id', aggfunc='first'
    ).reset_index()

    return result_df



def _assign_to_mask(mask,target,input):
    if np.any(mask):
        if np.ndim(input) == 0 or type(input) == tuple or type(input) == str:
            input = [input]*mask.sum()
        ids = np.arange(len(target))[mask]
        counter = 0
        for i in ids:
            target[i] = input[counter]
            counter += 1

def format_composition(composition):
    formatted_motifs = []
    for motif in composition:
        # Replace 'c' with 'b' and 'f', 'p' with 'u'
        motif_string = str(motif).replace('c','b').replace('f','u').replace('p','u')
        motif_string = "s_{"+str(motif_string)+"}"
        formatted_motifs.append(motif_string)
    return fr"$({', '.join(formatted_motifs)})$"

class BasisFunctions(ABC):

    def __init__(self, n_basis):
        self.n_basis = n_basis

    @abstractmethod
    def compute(self, x, x_range):
        pass

class BSplineBasisFunctions(BasisFunctions):

    def __init__(self,n_basis, k=3, include_bias=True, include_linear=False):
        super().__init__(n_basis)
        self.k = k
        self.include_bias = include_bias
        self.include_linear = include_linear
    
    def compute(self,X,X_ranges):

        categorical_features_indices = [i for i in range(X.shape[1]) if not isinstance(X_ranges[i],tuple)]

        if self.include_bias:
            n_b_basis = self.n_basis - 1
        else:
            n_b_basis = self.n_basis

        if self.include_linear:
            n_b_basis = n_b_basis - 1

        def singleton_vector(n,k):
            vector = np.zeros(n)
            vector[k] = 1
            return vector
        
        n_features = X.shape[1]
        B_list = []

        for feature_index in range(n_features):

            if feature_index in categorical_features_indices:
                continue

            x_range = X_ranges[feature_index]

            shape_knots = np.r_[[x_range[0]]*self.k,np.linspace(x_range[0], x_range[1], n_b_basis-self.k+1),[x_range[1]]*self.k]
       
            bsplines = [BSpline(shape_knots,singleton_vector(n_b_basis,k_index),k=self.k,extrapolate=False) for k_index in range(n_b_basis)]
            
            X_i = X[:,feature_index].flatten()
            # bspline_basis_per_sample = [BSpline(shape_knots,singleton_vector(n_b_basis,k_index),k=self.k,extrapolate=False)(X.flatten()) for k_index in range(n_b_basis)]
            bspline_basis_per_sample = [bspline(X_i) for bspline in bsplines]
            # fill na values with 0
            final_list = []
            for i, values in enumerate(bspline_basis_per_sample):
                below = X_i <= x_range[0]
                above = X_i >= x_range[1]
                values[below] = bsplines[i](x_range[0])
                values[above] = bsplines[i](x_range[1])
                # print(X.flatten())
                # print(values)
                if np.any(np.isnan(values)):
                    print(f'Nan values found for basis function {i}')
                    print(f'X: {X_i}')
                    print(f'Values: {values}')
                final_list.append(values)

            if self.include_bias:
                # add the constant basis function
                final_list.append(np.ones_like(X_i))
            
            if self.include_linear:
                # add the linear basis function
                values = X_i
                below = X_i <= x_range[0]
                above = X_i >= x_range[1]
                values[below] = x_range[0]
                values[above] = x_range[1]
                final_list.append(values)


            B_i = np.stack(final_list, axis=1) # shape (n_samples, n_basis)
            B_list.append(B_i)

        B = np.stack(B_list, axis=1) # shape (n_samples, n_cont_features, n_basis)
        return B
    
class OneHotBasisFunctions():

    def __init__(self):
        pass
    
    def compute(self,V,V_ranges):
        """
        Args: 
        V: a numpy array of shape (n_samples, n_features)
        V_ranges: a list of tuples or sets representing ranges of the features
        """
        categorical_features_indices = [i for i in range(len(V_ranges)) if not isinstance(V_ranges[i],tuple)]
        B_list = []
        n_unique_values_dict = {}
        for feature_index in categorical_features_indices:
            
            V_i = V[:,feature_index].flatten()
            unique_values = np.unique(V_i)

            # Check whether all unique values are in the appropriate range
            unique_values_set = set(unique_values)
            V_i_range = V_ranges[feature_index]
            if not unique_values_set.issubset(V_i_range):
                raise ValueError(f'Unique values {unique_values} for feature {feature_index} are not in the range {V_i_range}. \
                                 They were either not availalbe during training or do not belong to the appropriate leaves')
            
            V_i_range_list = sorted(list(V_i_range))
    
            n_unique_values = len(V_i_range_list)

            B_i = np.zeros((V_i.shape[0],n_unique_values))
            for i, value in enumerate(V_i_range_list):
                B_i[V_i == value,i] = 1
            B_list.append(B_i)

            n_unique_values_dict[feature_index] = n_unique_values
        
        B = np.concatenate(B_list, axis=1) # shape (n_samples, sum(n_unique_values))
        return B, n_unique_values_dict

    def compute_single(self,V,V_ranges,feature_index):
        """
        Args: 
        V: a numpy array of shape (n_samples, n_features)
        V_ranges: a list of tuples or sets representing ranges of the features
        """
        categorical_features_indices = [i for i in range(len(V_ranges)) if not isinstance(V_ranges[i],tuple)]
        
        if feature_index not in categorical_features_indices:
            raise ValueError(f'Feature {feature_index} is not a categorical feature')
        
        B_list = []
        V_i = V[:,feature_index].flatten()
        unique_values = np.unique(V_i)

        # Check whether all unique values are in the appropriate range
        unique_values_set = set(unique_values)
        V_i_range = V_ranges[feature_index]
        if not unique_values_set.issubset(V_i_range):
            raise ValueError(f'Unique values {unique_values} for feature {feature_index} are not in the range {V_i_range}. \
                                They were either not availalbe during training or do not belong to the appropriate leaves')
        
        V_i_range_list = sorted(list(V_i_range))

        n_unique_values = len(V_i_range_list)

        B_i = np.zeros((V_i.shape[0],n_unique_values))
        for i, value in enumerate(V_i_range_list):
            B_i[V_i == value,i] = 1

        return B_i    
    
class SemanticRepresentation:

    def __init__(self,t_range,composition,coordinates_finite_composition,derivative_start,derivative_end,properties_infinite_motif,second_derivative_end=None):
        self.t_range = t_range
        self.composition = composition
        self.coordinates_finite_composition = coordinates_finite_composition
        self.derivative_start = derivative_start
        self.derivative_end = derivative_end
        self.properties_infinite_motif = properties_infinite_motif
        self.second_derivative_end = second_derivative_end

    def __repr__(self):
        return f"""Composition: {self.composition}
Coordinates:
{self.coordinates_finite_composition}
Derivative at start: {self.derivative_start}
Derivative at end: {self.derivative_end}
Properties of infinite motif:
{self.properties_infinite_motif}
Second derivative at end: {self.second_derivative_end}"""
    
    def copy(self):
        return SemanticRepresentation(self.t_range,self.composition,self.coordinates_finite_composition,self.derivative_start,self.derivative_end,self.properties_infinite_motif,self.second_derivative_end)


def create_full_composition_library(max_length,is_infinite, simplified=False):

    motif_succession_rules = {
        '+-':['--','++'],
        '-+':['--','++'],
        '--':['-+'],
        '++':['+-']
    }

    motif_infinite_types = {
        '++':['f'],
        '+-':['p','h'],
        '-+':['f','h'],
        '--':['f']
    }

    all_compositions = []
    # dfs graph search algorithm
    def dfs(current_composition):

        if len(current_composition) > 0: # We do not add empty composition
            if is_infinite and current_composition[-1][2] != 'c':
                all_compositions.append(current_composition)
                return # The last motif is infinite, we cannot add more motifs
            elif not is_infinite:
                all_compositions.append(current_composition)
            # If the is_infinite but the last motif is finite, it's not a valid composition, so we do not add it to the list

        if len(current_composition) == max_length:
            return

        def expand(new_motif):
            if is_infinite:
                # We can make it a final motif by adding an infinite extension
                for infinite_extension in motif_infinite_types[new_motif]:
                    dfs(current_composition.copy() + [new_motif + infinite_extension])
                # We can also add a finite extension if there is still space
                if len(current_composition) < max_length-1:
                    dfs(current_composition.copy() + [new_motif + 'c'])
            else:
                dfs(current_composition.copy() + [new_motif + 'c'])

        if len(current_composition) == 0:
            for new_motif in ['+-','--','-+','++']:
                expand(new_motif)
        else:
            for new_motif in motif_succession_rules[current_composition[-1][0:2]]:
                expand(new_motif)
           
    dfs([])

    def is_simple(composition):
        for i in range(2,len(composition)):
            if composition[i][:2] == composition[i-2][:2]:
                return False
        return True

    if simplified:
        all_compositions = [composition for composition in all_compositions if is_simple(composition)]

    return all_compositions



class CompositionMap:

    def __init__(self,decision_tree_classifier,compositions):
        """
        Args:
        decision_tree_classifier: a DecisionTreeClassifier object. Predicts the index of the composition
        compositions: a list of compositions
        """

        self.decision_tree_classifier = decision_tree_classifier
        self.compositions = compositions

    def predict(self,X,reduce=False, include_indices=False):

        # If the input is a single sample, we need to reshape it
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        else:
            reduce = False
        
        composition_index = self.decision_tree_classifier.predict(X) # shape (n_samples,)
        compositions = [self.compositions[index] for index in composition_index] # shape (n_samples,)

        if len(compositions) == 1 and reduce:
            compositions = compositions[0]
            composition_index = composition_index[0]

        if include_indices:
            return compositions, composition_index
        else:
            return compositions
        
    def print(self):
        self.decision_tree_classifier.print_tree()

    def get_specific_feature_ranges(self,X_ranges,composition_index):
        """
        Get the range of the features for a specific composition

        Args:
        composition_index: an integer

        Returns:
        a list of tuples of floats representing the range of the features
        """
        return self.decision_tree_classifier.get_updated_feature_ranges(X_ranges)[composition_index]



class SemanticPredictor:

    def __init__(self,compostion_map,property_maps,t_range):
        """
        Args:
        compostion_map: a CompositionMap object
        property_maps: a dictionary with keys (composition: tuple of strings) and values of type SinglePropertyMap
        t_range: a tuple of floats representing the range of the time variable
        """
        self.composition_map = compostion_map
        self.property_maps = property_maps
        self.t_range = t_range

    def predict(self,V,reduce=True):
        """
        Predict the semantic representation

        Args:
        V: a numpy array of shape (batch_size, n_features)

        Returns:
        a list of SemanticRepresentation objects
        """
        if len(V.shape) == 1:
            V = V.reshape(1,-1)
        else:
            reduce = False

        compositions, composition_indices = self.composition_map.predict(V,include_indices=True)

        unique_composition_indices = np.unique(composition_indices)
        
        semantic_representations = np.empty(len(compositions),dtype=object)

        for composition_index in unique_composition_indices:
            mask = composition_indices == composition_index
            V_filtered = V[mask]
            composition = tuple(self.composition_map.compositions[composition_index])
            property_map = self.property_maps[composition]

            transition_points = property_map.predict_all_transition_points(V_filtered,reduce=False)
            derivative_start = property_map.predict_derivative(V_filtered,'start',1)
            derivative_end = property_map.predict_derivative(V_filtered,'end',1)
            second_derivative_end = property_map.predict_derivative(V_filtered,'end',2)
            properties_infinite_motif = property_map.predict_all_infinite_motif_properties(V_filtered)

            new_semantic_representations = []
            for j in range(V_filtered.shape[0]):
                new_semantic_representations.append(SemanticRepresentation(self.t_range,composition,transition_points[j],derivative_start[j],derivative_end[j],properties_infinite_motif[j],second_derivative_end[j]))

            _assign_to_mask(mask,semantic_representations, new_semantic_representations)

        if reduce:
            semantic_representations = semantic_representations[0]
        
        return semantic_representations
    

class PSODE:

    def __init__(self,
                 t_range,
                 M,
                 basis_functions,
                 composition_libraries,
                 seed=0,
                 opt_config={},
                 dt_config={},
                 x0_dict={},
                 verbose=False,
                 X_ranges=None,
                 categorical_features_indices=[],
                 feature_names=None):
        """
        Args:
        t_range: a tuple of floats representing the range of the time variable
        M: an integer representing the number of dimensions of the trajectory
        basis_functions: a list of M BasisFunctions objects, one for each dimension of the trajectory
        composition_liberaries: a list of lists of tuples of strings representing the compositions for each dimension of the trajectory
        seed: an integer representing the seed for the random number generator
        opt_config: a dictionary with optimization configuration parameters
        dt_config: a dictionary with decision tree configuration parameters
        x0_dict: a dictionary with the initial condition configuration parameters. (m: k) means that the initial condition for the m-th trajectory is given by the k-th feature
                    can also be (m: f) where f is a float representing the initial condition---constant for all trajectories
        verbose: a boolean indicating whether to print information
        X_ranges: a list of tuples of floats representing the range of the static features
        categorical_features_indices: a list of integers representing the indices of the categorical features. \
            If not provided, the categorical features are assumed to be encoded as categories in the dataframe. If both are provided then \
            they need to be consistent.
        feature_names: a list of strings representing the names of the features. If not provided, the names are assumed to be the column names of the dataframe. \
            If both are provided then they need to be consistent.
        """

        self.single_psodes = [SinglePSODE(t_range=t_range,
                                          basis_functions=basis_functions[m],
                                          composition_library=composition_libraries[m],
                                          seed=seed,
                                          opt_config=opt_config,
                                          dt_config=dt_config,
                                          verbose=verbose,
                                          X_ranges=X_ranges,
                                          x0_included=(m in x0_dict),
                                          x0_index = x0_dict.get(m,-1),
                                          categorical_features_indices=categorical_features_indices) for m in range(M)]
        self.seed = seed
        self.composition_maps = {}
        self.categorical_features_indices = categorical_features_indices
        self.categorical_features_categories = {}
        self.feature_names = feature_names


    def _transform_V_df_to_numpy(self,V, task):

        if task == 'fit' and self.feature_names is not None:
            if not np.array_equal(V.columns,self.feature_names):
                raise ValueError('The features of the dataframe are not the same as the one passed to the constructor.')

        V = V.copy()
        # Check if any of the features in the dataframe are categorical
        df_categorical = [i for i, col in enumerate(V.columns) if V[col].dtype.name == 'category']

        if len(df_categorical) > 0 and len(self.categorical_features_indices) > 0:
            # Check if the categorical features are the same
            if set(df_categorical) != set(self.categorical_features_indices):
                if task == 'fit':
                    raise ValueError('Categorical features are encoded in two incompatible ways. Please either encode the columns of \
                                        the dataframe as categorical or provide a list of categorical features indices in the constructor')
                elif task == 'predict':
                    raise ValueError('The categorical features in the dataframe are not the same as the ones used during training.')
        elif len(df_categorical) == 0:
            # That means that the categorical features are NOT encoded in the dataframe
            # We need to (possibly) encode them
            for i in self.categorical_features_indices:
                V.iloc[:,i] = V.iloc[:,i].astype('category')
        elif len(self.categorical_features_indices) == 0:
            # That means that the categorical features are encoded in the dataframe
            if task == 'fit':
                self.categorical_features_indices = df_categorical
            elif task == 'predict':
                raise ValueError('The categorical features in the dataframe are not the same as the ones used during training.')

        if task == 'fit':  
            self.feature_names = V.columns
            for i in self.categorical_features_indices:
                self.categorical_features_categories[i] = list(V.iloc[:,i].cat.categories)
        elif task == 'predict':
            if not np.array_equal(V.columns,self.feature_names):
                raise ValueError('The features of the dataframe are not the same as the ones used during training.')
            for i in self.categorical_features_indices:
                V.iloc[:,i] = V.iloc[:,i].astype('category')
                V.iloc[:,i] = V.iloc[:,i].cat.set_categories(self.categorical_features_categories[i])
       
        # At this point V should have all the categorical features encoded in the same way as during training
        # Convert categories to integers and then to floats
        for i in self.categorical_features_indices:
            column_name = V.columns[i]
            V[column_name] = V[column_name].cat.codes
            V[column_name] = V[column_name].astype('float32')

        return V.to_numpy()
     
    def fit(self,V,T,Y, composition_maps={}, all_soft_constraints={}):
        """
        DOES NOT WORK FOR DIFFERENT NUMBERS OF MEASUREMENTS

        Fit the PSODE model to the data

        Args:
        V: a numpy array (batch_size, n_features), categorical features should encoded by OrdinalEncoder or pandas dataframe \
            where categorical features are encoded as categories. Alternatively the categorical features can be specified in the constructor.
        T: a list of numpy arrays, each of shape N_d
        Y: a list of numpy arrays, each of shape N_d x M
        composition_maps: a dictionary of (m: CompositionMap), where m is the index of dimension of the trajectory
        """
        if isinstance(V,pd.DataFrame):
            V = self._transform_V_df_to_numpy(V, 'fit')
           
        for m, single_psode in enumerate(self.single_psodes):
            single_psode.feature_names = self.feature_names
            single_psode.categorical_features_categories = self.categorical_features_categories
            single_psode.categorical_features_indices = self.categorical_features_indices
            Y_m = np.stack([Y[d][:,m] for d in range(len(Y))],axis=0)
            T_m = np.stack(T,axis=0)
            V_m = V
            if m in composition_maps:
                composition_map = composition_maps[m]
            else:
                composition_map = None
            if m in all_soft_constraints:
                soft_constraints = all_soft_constraints[m]
            else:
                soft_constraints = None
            single_psode.fit(V_m,T_m,Y_m,composition_map,soft_constraints)
        
        self.composition_maps = {m: single_psode.composition_map for m, single_psode in enumerate(self.single_psodes)}

    def fit_composition_maps(self,V,T,Y, composition_scores_dict={}):
        """
        Fit the composition maps to the data

        Args:
        V: a numpy array (batch_size, n_features), categorical features should encoded by OrdinalEncoder or pandas dataframe \
            where categorical features are encoded as categories. Alternatively the categorical features can be specified in the constructor.
        T: a list of numpy arrays, each of shape N_d
        Y: a list of numpy arrays, each of shape N_d x M
        """
        composition_maps = {}
        if isinstance(V,pd.DataFrame):
            V = self._transform_V_df_to_numpy(V, 'fit')
        for m, single_psode in enumerate(self.single_psodes):
            single_psode.feature_names = self.feature_names
            single_psode.categorical_features_categories = self.categorical_features_categories
            single_psode.categorical_features_indices = self.categorical_features_indices
            Y_m = np.stack([Y[d][:,m] for d in range(len(Y))],axis=0)
            T_m = np.stack(T,axis=0)
            V_m = V
            if m in composition_scores_dict:
                composition_scores = composition_scores_dict[m]
            else:
                composition_scores = None
            composition_map = single_psode.fit_composition_map(V_m,T_m,Y_m, composition_scores=composition_scores)
            composition_maps[m] = composition_map
        
        return composition_maps

    def predict(self,V,T):
        if isinstance(V,pd.DataFrame):
            V = self._transform_V_df_to_numpy(V, 'predict')
        # return self.predict_raw(V,T)
        return np.stack([single_psode.predict(V,T) for single_psode in self.single_psodes],axis=2)
        

    def predict_raw(self,V,T):
        """
        DOES NOT WORK FOR DIFFERENT NUMBERS OF MEASUREMENTS

        Args:
        V: a numpy array of shape (batch_size, n_features), categorical features should encoded by OrdinalEncoder or similar
        T: a list of numpy arrays, each of shape N_d

        Returns:
        a numpy aray of shape (batch_size, N, M)
        """
        if isinstance(V,pd.DataFrame):
            V = self._transform_V_df_to_numpy(V, 'predict')
        return np.stack([single_psode.predict_raw(V,T) for single_psode in self.single_psodes],axis=2)
        

class SinglePSODE:

    def __init__(self,
                 t_range,
                 basis_functions,
                 composition_library,
                 seed=0,
                 opt_config={},
                 dt_config={},
                 verbose=False,
                 X_ranges=None,
                 x0_included=False,
                 x0_index=-1,
                 categorical_features_indices=[]
             ):

        self.config = self._get_updated_opt_config(opt_config)
        self.dt_config = self._get_updated_dt_config(dt_config)
        print(self.config)
        
        self.t_range = t_range
        self.X_ranges = X_ranges
        self.basis_functions = basis_functions
        self.composition_library = composition_library
        self.categorical_features_indices = categorical_features_indices
        
        self.verbose = verbose

        self.config['t_range'] = t_range
        self.config['n_basis_functions'] = basis_functions.n_basis
        self.config['seed'] = seed
        self.config['x0_included'] = x0_included
        self.config['x0_index'] = x0_index
        self.x0_included = x0_included

        self.composition_scores = None
        self.composition_map = None
        self.torch_models = None

        self.lightning_accelerator = utils.get_lightning_accelerator(self.config['device'])
        self.torch_device = utils.get_torch_device(self.config['device'])

        self.one_hot_basis_functions = OneHotBasisFunctions()

        self.feature_names = None
        self.categorical_features_mappings = {}
        

    def _get_updated_opt_config(self,opt_config):
        config = self._get_default_opt_config()
        for key, value in opt_config.items():
            config[key] = value
        return config
    
    def _get_updated_dt_config(self,dt_config):
        config = self._get_default_dt_config()
        for key, value in dt_config.items():
            config[key] = value
        return config
    

    def _get_tensors(self,*args):
        """
        Convert numpy arrays to torch tensors

        Args:
        args: numpy arrays

        Returns:
        tuple of torch tensors
        """
        return tuple([torch.tensor(arg, dtype=torch.float32, device=self.torch_device) for arg in args])
    
    # def _finite_x0_range(self,prev_x0,next_x0):
    #     """
    #     Get the range of the initial condition

    #     Args:
    #     X0: initial condition numpy array of shape (batch_size, 1)

    #     Returns:
    #     tuple of the range of the initial condition
    #     """
    #     if prev_x0 is None:
    #         prev_x0 = -np.inf
    #     if prev_x0 > -np.inf:
    #         x0_range_0 = prev_x0
    #     elif self.x0_range is not None:
    #         x0_range_0 = self.x0_range[0]

    #     if next_x0 is None:
    #         next_x0 = np.inf
    #     if next_x0 < np.inf:
    #         x0_range_1 = next_x0
    #     elif self.x0_range is not None:
    #         x0_range_1 = self.x0_range[1]

    #     x0_range = (x0_range_0, x0_range_1)
    #     return x0_range
    
    def _get_global_X_ranges(self,X):
        """
        Get the range of the initial condition

        Args:
        X: static feature numpy array of shape (batch_size, n_features)

        Returns:
        tuple of the range of the initial condition
        """
        X_ranges = []
        if self.X_ranges is None:
            for i in range(X.shape[1]):
                if i in self.categorical_features_indices:
                    max_category = int(X[:,i].max())
                    X_ranges.append(set(range(max_category+1)))
                else:
                    X_ranges.append((X[:,i].min(),X[:,i].max()))
            self.X_ranges = X_ranges
        return self.X_ranges
    
    def _get_train_val_indices(self,n_samples):
        """
        Get the indices for the training and validation sets

        Args:
        n_samples: number of samples

        Returns:
        tuple of numpy arrays with the indices
        """
        np_gen = np.random.default_rng(self.config['seed'])
        train_indices = np_gen.choice(n_samples, int(0.8*n_samples), replace=False)
        val_indices = np.setdiff1d(np.arange(n_samples), train_indices)
        return train_indices, val_indices
    

    def _compute_composition_scores(self,V,T,Y):

        n_samples = V.shape[0]

        train_on_whole_trajectory = self.dt_config['train_on_whole_trajectory']

        if self.config['x0_included']:
            if isinstance(self.config['x0_index'],int):
                x0 = V[:,self.config['x0_index']]
            elif isinstance(self.config['x0_index'],float):
                x0 = [self.config['x0_index']] * n_samples
            else:
                raise ValueError('x0_index should be an integer or a float')
        else:
            x0 = [None] * n_samples

        composition_ids = list(range(len(self.composition_library)))
        
        info_list = []
        for composition_id in composition_ids:
            samples = zip(x0.copy(),T.copy(),Y.copy())
            sample_ids = range(n_samples)
            composition = tuple(self.composition_library[composition_id])

            composition_id_list = copy.deepcopy([composition_id] * n_samples)
            composition_list = copy.deepcopy([composition] * n_samples)

            info = zip(samples,
                        composition_id_list,
                        composition_list,
                        [self.config['t_range']] * n_samples, 
                        [self.config['seed']] * n_samples, sample_ids, [train_on_whole_trajectory] * n_samples)
            info_list.append(info)
        all_infos = itertools.chain(*info_list)
        with multiprocessing.Pool() as p:
            composition_scores = list(tqdm(p.imap(process_sample_one_comp, all_infos), total=n_samples*len(composition_ids)))
        # Create a dataframe
        composition_scores_df = combine_dicts_to_dataframe(composition_scores)

        # Check if there are any NaN values
        if composition_scores_df.isnull().values.any():
            raise ValueError('NaN values found in the composition scores')

        # sort by id
        composition_scores_df = composition_scores_df.sort_values(by='id')

        # Remove id column
        composition_scores_df = composition_scores_df.drop(columns=['id'])



        # Reorder compositions
        composition_scores_df = composition_scores_df[composition_ids]

        composition_scores = composition_scores_df.values

        return composition_scores
    
    def save_composition_scores_df(self,folder='composition_scores',suffix=""):
        if not os.path.exists(folder):
            os.makedirs(folder)
        composition_scores_df = pd.DataFrame(self.composition_scores)
        if suffix != "":
            filename = f'composition_scores_{suffix}.csv'
        else:
            filename = 'composition_scores.csv'
        composition_scores_df.to_csv(os.path.join(folder,filename),index=False)
    
    def visualize_composition_scores_df(self, ax, monotonicty_scores=False, raw_scores=False):

        if self.composition_scores is None:
            raise ValueError('Composition map has not been fitted')

        if monotonicty_scores:
            data_to_plot, reduced_compositions, unique_reduced_compositions = self._get_monotonicity_scores(self.composition_scores)
        else:
            data_to_plot = self.composition_scores

        if not raw_scores:
            data_to_plot = utils.softmax_per_row(-data_to_plot)

        # Create the heatmap
        sns.heatmap(data_to_plot, ax=ax, annot=False, cmap='viridis', cbar_kws={'shrink': .5})

        # Set x-axis labels to be the column names

        if monotonicty_scores:
            x_labels = unique_reduced_compositions
        else:
            x_labels = [format_composition(column_name) for column_name in self.composition_library]
        ax.set_xticks(np.arange(len(x_labels)) + 0.5)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        # Set the aspect ratio to ensure a square-ish plot
        ax.set_aspect('auto')

    def _get_monotonicty_from_composition(self,composition):
        reduced_composition = ''
        for i in range(len(composition)):
            if i > 0:
                if composition[i][0] == composition[i-1][0]:
                    continue
            reduced_composition += composition[i][0]
        return reduced_composition

    def _get_monotonicity_from_compositions(self,compositions):
        reduced_compositions = []
        for composition in compositions:
            reduced_compositions.append(self._get_monotonicty_from_composition(composition))
        return reduced_compositions
    
    def _get_monotonicity_scores(self,composition_scores):

        reduced_compositions = self._get_monotonicity_from_compositions(self.composition_library)
        unique_reduced_compositions = np.unique(reduced_compositions)

        # For each unique reduced composition, get the minimum error
        monotonicity_scores = np.zeros((composition_scores.shape[0],len(unique_reduced_compositions)))
        for i, reduced_composition in enumerate(unique_reduced_compositions):
            mask = (np.array(reduced_compositions) == reduced_composition)
            errors = composition_scores[:,mask]
            monotonicity_scores[:,i] = errors.min(axis=1)

        return monotonicity_scores, reduced_compositions, unique_reduced_compositions


    def fit_composition_map(self,V,T,Y, composition_scores=None):
        
        if self.verbose:
            print(f"Fitting the composition map to the data")

        if composition_scores is None:
            composition_scores = self._compute_composition_scores(V,T,Y)

        self.composition_scores = composition_scores.copy()

        # Save composition scores as a csv
        # composition_scores_df = pd.DataFrame(composition_scores)
        # composition_scores_df.to_csv('composition_scores.csv',index=False)
        print(self.categorical_features_indices)
        dt = DecisionTreeClassifier(max_depth=self.dt_config['max_depth'],
                                    metric_name='error',
                                    offset=self.dt_config['min_relative_gain_to_split'],
                                    min_samples_leaf=self.dt_config['min_samples_leaf'],
                                    categorical_features=self.categorical_features_indices)

        # monotonicity_scores, reduced_compositions, unique_reduced_compositions = self._get_monotonicity_scores(composition_scores)

        # print(unique_reduced_compositions)
        # print(reduced_compositions)
        
        # # monotonicity_scores = utils.softmax_per_row(-monotonicity_scores)

        # dt = DecisionTreeClassifier(max_depth=3,metric_name='error',offset=self.composition_map_offset)

        # dt.fit(V,monotonicity_scores)

        # dt.print_tree()

        # print('Possible predictions:')

        # possible_reduced_composition_indices = dt.get_predictions_at_leaves()
        # print(possible_reduced_composition_indices)
        # possible_reduced_compositions = [unique_reduced_compositions[index] for index in possible_reduced_composition_indices]
        # print(possible_reduced_compositions)

        # filtered_compositions = []
        # # Compositions that with the corresponding reduced compositions
        # for red_comp, comp in zip(reduced_compositions, self.composition_library):
        #     if red_comp in possible_reduced_compositions:
        #         filtered_compositions.append(tuple(comp))

        # print(filtered_compositions)

        # # Create mask
        # mask = np.zeros(len(self.composition_library),dtype=bool)
        # for i, comp in enumerate(self.composition_library):
        #     if tuple(comp) in filtered_compositions:
        #         mask[i] = True
        
        # composition_scores = composition_scores[:,mask]

        

        # composition_scores = utils.softmax_per_row(-composition_scores)

        # dt.continue_fit(V,composition_scores)
        composition_lengths = np.array([len(comp) for comp in self.composition_library]).reshape(1,-1)

        composition_scores = composition_scores.copy() * (1 + composition_lengths * self.dt_config['relative_motif_cost'])
        
        if self.dt_config['tune_depth']:

            # Divde v into train and validation sets and fit the decision tree
            train_indices, val_indices = self._get_train_val_indices(V.shape[0])

            V_train = V[train_indices]
            composition_scores_train = composition_scores[train_indices]

            V_val = V[val_indices]
            composition_scores_val = composition_scores[val_indices]

            results_of_tuning = {}

            for tested_depth in range(0,self.dt_config['max_depth']+1):
                dt = DecisionTreeClassifier(max_depth=tested_depth,
                                            metric_name='error',
                                            offset=self.dt_config['min_relative_gain_to_split'],
                                            min_samples_leaf=self.dt_config['min_samples_leaf'],
                                            categorical_features=self.categorical_features_indices)
                dt.fit(V_train,composition_scores_train)
                pred_train_indices = dt.predict(V_train)
                pred_val_indices = dt.predict(V_val)
                train_loss = np.mean([composition_scores_train[i,pred_train_indices[i]] - np.min(composition_scores_train[i,:]) for i in range(len(pred_train_indices))])
                val_loss = np.mean([composition_scores_val[i,pred_val_indices[i]] - np.min(composition_scores_val[i,:]) for i in range(len(pred_val_indices))])
                print(f"Depth: {tested_depth}, Train loss: {train_loss}, Val loss: {val_loss}")
                results_of_tuning[tested_depth] = val_loss
            
            best_depth = min(results_of_tuning, key=results_of_tuning.get)

            if self.verbose:
                print(f'Best depth: {best_depth}')
        
        else:

            best_depth = self.dt_config['max_depth']

        # Fit the decision tree with the best depth on the whole dataset

        dt = DecisionTreeClassifier(max_depth=best_depth,
                                    metric_name='error',
                                    offset=self.dt_config['min_relative_gain_to_split'],
                                    min_samples_leaf=self.dt_config['min_samples_leaf'],
                                    categorical_features=self.categorical_features_indices)

        dt.fit(V, composition_scores)

        composition_map = CompositionMap(dt, self.composition_library)
        
        if self.verbose:
            print('Composition map')
            composition_map.print()
        
        return composition_map
    
    def _construct_histograms(self,V, specific_V_ranges):

        histograms = []
        for feature_index in range(V.shape[1]):
            if feature_index in self.categorical_features_indices:
                possible_classes = sorted(list(specific_V_ranges[feature_index]))
                histogram = {}
                for value in possible_classes:
                    histogram[value] = np.sum(V[:,feature_index] == value)
            else:
                histogram = np.histogram(V[:,feature_index], bins=20)
            histograms.append(histogram)
        return histograms

    def fit_property_maps(self,V,T,Y,composition_map,soft_constraints=None):

        if self.verbose:
            print(f'Fitting the property maps to the data')
        
        val_loss = 0
        n_val_samples = 0
        torch_models = []
        property_maps = {}

        X_ranges = self._get_global_X_ranges(V)
    
        if self.verbose:
            print("Global V ranges:")
            print(X_ranges)

        if self.x0_included:
            if isinstance(self.config['x0_index'],int):
                X0 = V[:,[self.config['x0_index']]]
            elif isinstance(self.config['x0_index'],float):
                X0 = np.tile(self.config['x0_index'], (V.shape[0], 1))
            else:
                raise ValueError('x0_index must be an integer or a float')
          
        predicted_compositions, predicted_composition_indices = composition_map.predict(V, include_indices=True)

        unique_composition_indices = np.unique(predicted_composition_indices)

        property_map_extractor = PropertyMapExtractor(self.config,self.basis_functions,self.one_hot_basis_functions)

        for composition_index in unique_composition_indices:
            mask = predicted_composition_indices == composition_index
            # mask = [c == composition_index for c in predicted_composition_indices]
            V_numpy_filtered = V[mask]
            T_numpy_filtered = T[mask]
            Y_numpy_filtered = Y[mask]

            specific_V_ranges = composition_map.get_specific_feature_ranges(X_ranges,composition_index)
            print(f"Composition {composition_index} has specific ranges {specific_V_ranges}")

            B_numpy_filtered = self.basis_functions.compute(V_numpy_filtered,specific_V_ranges)

            V_filtered, B_filtered, T_filtered, Y_filtered = self._get_tensors(V_numpy_filtered,B_numpy_filtered,T_numpy_filtered,Y_numpy_filtered)

            if len(self.categorical_features_indices) > 0:
                B_cat_numpy_filtered, cat_n_unique_dict = self.one_hot_basis_functions.compute(V_numpy_filtered,specific_V_ranges)
                B_cat_filtered = self._get_tensors(B_cat_numpy_filtered)[0]
            else:
                B_cat_filtered = torch.zeros_like(V_filtered)[:,[0]]
                cat_n_unique_dict = {}

            if self.x0_included:
                X0_numpy_filtered = X0[mask]
                X0_tensor = self._get_tensors(X0_numpy_filtered)[0]


            n_samples = V_filtered.shape[0]
            n_features = V_filtered.shape[1]

            train_indices, val_indices = self._get_train_val_indices(n_samples)

            if self.x0_included:
                train_dataset = torch.utils.data.TensorDataset(V_filtered[train_indices], B_filtered[train_indices], T_filtered[train_indices], Y_filtered[train_indices], X0_tensor[train_indices], B_cat_filtered[train_indices])
                val_dataset = torch.utils.data.TensorDataset(V_filtered[val_indices], B_filtered[val_indices], T_filtered[val_indices], Y_filtered[val_indices], X0_tensor[val_indices], B_cat_filtered[val_indices])
            else:
                train_dataset = torch.utils.data.TensorDataset(V_filtered[train_indices], B_filtered[train_indices], T_filtered[train_indices], Y_filtered[train_indices], B_cat_filtered[train_indices])
                val_dataset = torch.utils.data.TensorDataset(V_filtered[val_indices], B_filtered[val_indices], T_filtered[val_indices], Y_filtered[val_indices], B_cat_filtered[val_indices])
            
            composition_config = self.config.copy()
            composition = tuple(self.composition_library[composition_index])

            composition_config['soft_constraint'] = None
            if soft_constraints is not None:
                if composition in soft_constraints:
                    composition_config['soft_constraint'] = soft_constraints[composition]

            composition_config['composition'] = composition
            composition_config['n_features'] = n_features
            composition_config['categorical_features_indices'] = self.categorical_features_indices
            composition_config['cat_n_unique_dict'] = cat_n_unique_dict

            tuning = (self.config['n_tune'] > 0)
            val_loss_per_branch, model = self._fit_composition(composition_config,composition,train_dataset,val_dataset,tuning=tuning)
            val_loss += val_loss_per_branch * len(val_indices)
            n_val_samples += len(val_indices)
            torch_models.append(model.model)
            histograms = self._construct_histograms(V_numpy_filtered, specific_V_ranges)
            
            property_maps[composition] = property_map_extractor.construct_single_property_map(composition,model.model,specific_V_ranges, histograms)

            if self.verbose:
                print(f"Validation loss for composition {composition}: {val_loss_per_branch}")
        if self.verbose:
            print(f'All property maps fitted')
    
        return property_maps, val_loss/n_val_samples, torch_models
    

    def _validate(self,V):
        """
        Validate the input features

        Args:
        V: a numpy array of shape (batch_size, n_features)
        """

        # Check if the categorical features are encoded as integers and whether all values are available
        for feature_index in self.categorical_features_indices:
            V_i = V[:,feature_index].flatten()
            # Convert to integers
            V_i_int = V_i.astype(int)
            # Check if the difference is 0
            if not np.allclose(V_i - V_i_int,0):
                raise ValueError(f'Feature {feature_index} is categorical but not encoded as integers')
            max_value = np.max(V_i_int)
            # Check if all values between 0 and max_value are present
            if not np.all(np.isin(np.arange(max_value+1),V_i_int)):
                raise ValueError(f'Feature {feature_index} is categorical but not all values are present')


    def fit(self,V,T,Y,composition_map=None,soft_constraints=None):
        """
        Fit the model to the data

        Args:
        V: static feature tensor of shape (batch_size, n_features)
        T: time tensor of shape (batch_size, n_measurements)
        Y: output tensor of shape (batch_size, n_measurements)

        Returns:
        """

        self._validate(V)

        self.org_V = V
        self.org_T = T
        self.org_Y = Y

        self.config['categorical_features_indices'] = self.categorical_features_indices
        self.config['categorical_features_categories'] = self.categorical_features_categories
        self.config['feature_names'] = self.feature_names

        self.torch_models = None
        self._get_global_X_ranges(V)

        if composition_map is None:
            composition_map = self.fit_composition_map(V,T,Y)
        else:
            if self.verbose:
                print('Using the provided composition map')


        property_maps, loss, torch_models = self.fit_property_maps(V,T,Y,composition_map,soft_constraints=soft_constraints)
        # semantic_predictor = SemanticPredictor(composition_map,property_maps,self.t_range)
        # self.semantic_predictor = semantic_predictor
        self.torch_models = torch_models
        semantic_predictor = SemanticPredictor(composition_map,property_maps,self.t_range)
        self.semantic_predictor = semantic_predictor

        print(f"Semantic predictor fitted with validation loss: {loss}")

    def predict_raw(self,V,T):

        if self.semantic_predictor is None:
            raise ValueError('Model has not been fitted yet')

        semantic_representations = self.semantic_predictor.predict(V, reduce=False)

        y_pred_list = []
        for i, semantic_representation in enumerate(semantic_representations):
            predictive_model = ApproximatePredictiveModel(semantic_representation)
            y_pred_list.append(predictive_model.forward(T[i]))
        
        results = np.stack(y_pred_list,axis=0)

        return results

    def predict(self,V,T):

        if self.semantic_predictor is None:
            raise ValueError('Model has not been fitted yet')
        
        semantic_representations = self.semantic_predictor.predict(V, reduce=False)

        y_pred_list = []
        for i, semantic_representation in enumerate(semantic_representations):
            predictive_model = PredictiveModel(semantic_representation)
            if predictive_model.converged:
                y_pred_list.append(predictive_model.predict(T[i]))
            else:
                y_pred_list.append(self.predict_raw(V[[i],:],[T[i]])[0])

        results = np.stack(y_pred_list,axis=0)
        
        return results

        # n_samples = X.shape[0]
        # n_measurements = T.shape[1]

        # Y_pred = torch.zeros(n_samples, n_measurements)

        # X_range = self._get_global_X_ranges(X)
        # B = self.basis_functions.compute(X,X_range)

        # X_tensor, B_tensor, T_tensor, = self._get_tensors(X,B,T)
        # if self.x0_included:
        #     X0_tensor = self._get_tensors(X0)[0]
        # torch_model = self.torch_models[0]
        # torch_model.eval()
        # with torch.no_grad():
        #     if self.x0_included:
        #         Y_pred = torch_model(X_tensor, B_tensor, T_tensor, X0=X0_tensor).detach().cpu()
        #     else:
        #         Y_pred = torch_model(X_tensor, B_tensor, T_tensor).detach().cpu()
        
        # results = Y_pred.numpy()


        # if is_x0_scalar:
        #     results = results[0]
        # if is_T_scalar:
        #     results = results[0]
        
    
    # def extract_shape_function_of_transition_point(self, ind, coordinate, feature_index):

    #     def shape_function(x):
    #         if self.x0_included and feature_index == self.config['x0_index']:
    #             X0_tensor = self._get_tensors(x.reshape(-1,1))[0]
    #         else:
    #             X0_tensor = None
    #         X = np.zeros((len(x),len(self.X_ranges)))
    #         X[:,feature_index] = x
    #         B = self.basis_functions.compute(X,self.X_ranges)
    #         b = B[:,feature_index,:]
    #         b_tensor = self._get_tensors(b)[0]
    #         torch_model = self.torch_models[0]
    #         torch_model.eval()
    #         with torch.no_grad():
    #             pred = torch_model.extract_shape_function_of_transition_point(ind,coordinate,feature_index,b_tensor,X0_tensor).detach().cpu()
    #         return pred.numpy()
        
    #     return shape_function
    
    # def extract_bias_of_transition_point(self, ind, coordinate):

    #     torch_model = self.torch_models[0]
    #     torch_model.eval()
    #     with torch.no_grad():
    #         pred = torch_model.extract_bias_of_transition_point(ind,coordinate).detach().cpu()
    #     return pred.numpy()
        
    # def extract_shape_function_of_infinite_property(self, ind, feature_index):

    #     def shape_function(x):
    #         if self.x0_included and feature_index == self.config['x0_index']:
    #             X0_tensor = self._get_tensors(x.reshape(-1,1))[0]
    #         else:
    #             X0_tensor = None
    #         X = np.zeros((len(x),len(self.X_ranges)))
    #         X[:,feature_index] = x
    #         B = self.basis_functions.compute(X,self.X_ranges)
    #         b = B[:,feature_index,:]
    #         b_tensor = self._get_tensors(b)[0]
    #         torch_model = self.torch_models[0]
    #         torch_model.eval()
    #         with torch.no_grad():
    #             pred = torch_model.extract_shape_function_of_infinite_property(ind,feature_index,b_tensor,X0_tensor).detach().cpu()
    #         return pred.numpy()
        
    #     return shape_function
    
    # def extract_bias_of_infinite_property(self, ind):

    #     torch_model = self.torch_models[0]
    #     torch_model.eval()
    #     with torch.no_grad():
    #         pred = torch_model.extract_bias_of_infinite_property(ind).detach().cpu()
    #     return pred.numpy()
    
    def visualize_gam_of_infinite_property(self,ind,axs=None, feature_names=None):

        if axs is None:
            fig, axs = plt.subplots(1,len(self.X_ranges)+1,figsize=(15,2.5))

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.X_ranges))]


        bias = self.extract_bias_of_infinite_property(ind)

        
        for i, X_range in enumerate(self.X_ranges):
            x = np.linspace(X_range[0],X_range[1],100)
            y = self.extract_shape_function_of_infinite_property(ind,i)(x)
            y_mean = np.mean(y,axis=0)
            y = y - y_mean
            bias += y_mean
            axs[i].plot(x,y,label=f'Feature {i}')
            axs[i].set_title(feature_names[i])
            axs[i].set_ylabel('Shape function')
            curr_ylim = axs[i].get_ylim()
            ylim_max = max(curr_ylim[1],0.01)
            ylim_min = min(curr_ylim[0],-0.01)
            axs[i].set_ylim(ylim_min,ylim_max)

        
        axs[len(self.X_ranges)].plot(np.linspace(0,1,100),bias*np.ones(100))
        axs[len(self.X_ranges)].set_title('Bias')

        plt.tight_layout()
    
    def visualize_gam_of_transition_point(self,ind,coordinate,axs=None, feature_names=None):

        if axs is None:
            fig, axs = plt.subplots(1,len(self.X_ranges)+1,figsize=(15,2.5))

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.X_ranges))]

        bias = self.extract_bias_of_transition_point(ind,coordinate)
        
        for i, X_range in enumerate(self.X_ranges):
            x = np.linspace(X_range[0],X_range[1],100)
            y = self.extract_shape_function_of_transition_point(ind,coordinate,i)(x)
            y_mean = np.mean(y,axis=0)
            y = y - y_mean
            bias += y_mean
            axs[i].plot(x,y,label=f'Feature {i}')
            axs[i].set_title(feature_names[i])
            axs[i].set_ylabel('Shape function')
            curr_ylim = axs[i].get_ylim()
            ylim_max = max(curr_ylim[1],0.01)
            ylim_min = min(curr_ylim[0],-0.01)
            axs[i].set_ylim(ylim_min,ylim_max)
        
        axs[len(self.X_ranges)].plot(np.linspace(0,1,100),bias*np.ones(100))
        axs[len(self.X_ranges)].set_title('Bias')

        plt.tight_layout()
    



        

    
    # def predict(self,X0,T):

    #     is_scalar = np.isscalar(X0)
    #     is_scalar_T = np.isscalar(T)

    #     T = np.atleast_1d(T)
    #     if len(T.shape) == 1:
    #         T = T.reshape(1,-1)

    #     X0 = np.atleast_1d(X0)
    #     if len(X0.shape) == 1:
    #         X0 = X0.reshape(-1,1)
        

    #     if self.semantic_predictor is None:
    #         raise ValueError('Model has not been fitted yet')
        
    #     semantic_representations = self.semantic_predictor.predict(X0)

    #     y_pred_list = []
    #     for i, semantic_representation in enumerate(semantic_representations):
    #         predictive_model = PredictiveModel(semantic_representation)
    #         if predictive_model.converged:
    #             y_pred_list.append(predictive_model.predict(T[i]))
    #         else:
    #             y_pred_list.append(self.predict_raw(X0[[i],:],T[[i],:])[0])

    #     results = np.stack(y_pred_list,axis=0)

    #     if is_scalar:
    #         results = results[0]
    #     if is_scalar_T:
    #         results = results[0]
        
    #     return results
       
    
    # def predict_from_semantic_representation(self, semantic_representation, t):
    #     predictive_model = PredictiveModel(semantic_representation)
    #     if predictive_model.converged:
    #         return predictive_model.predict(t)
    #     else:
    #         raise ValueError('Predictive model did not converge')

    def _fit_composition_objective(self,composition_config,composition,train_dataset,val_dataset,tune,parameters):

        for key, value in parameters.items():
            composition_config[key] = value
        litmodule = LitSketchODE(composition_config)
        assert litmodule.model.composition == composition

        # if self.torch_models is not None:
        #     litmodule.model.fixed_infinite_properties = self.torch_models[property_index].fixed_infinite_properties
            

        composition_timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        if self.verbose:
            print(f'Fitting the model to the data using the composition: {composition}')

       
        # tuner = Tuner(trainer)
        # lr_finder = tuner.lr_find(litmodule,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

        # suggested_lr = lr_finder.suggestion()
        # if suggested_lr is not None:
        #     litmodule.lr = lr_finder.suggestion()

        val_loss = np.inf

        # create callbacks
        best_val_checkpoint = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            filename='best_val',
            dirpath=f'./checkpoints/{composition_timestamp}'
        )
        # added early stopping callback, if validation loss does not improve over 10 epochs -> terminate training.
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10 if tune else 20,
            verbose=False,
            mode='min'
        )
        callback_ls = [best_val_checkpoint, early_stop_callback]

        trainer_dict = {
            'deterministic': True,
            'devices': 1,
            'enable_model_summary': False,
            'enable_progress_bar': self.verbose,
            'accelerator': self.lightning_accelerator,
            'max_epochs': self.config['n_epochs'] if not tune else self.config['n_epochs']//2,
            'check_val_every_n_epoch': 2,
            'log_every_n_steps': 1,
            'callbacks': callback_ls
        }
        trainer = L.Trainer(**trainer_dict)

        torch_gen = torch.Generator()
        torch_gen.manual_seed(self.config['seed'])

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, generator=torch_gen)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        
                
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING) 
        logging.getLogger('lightning').setLevel(0)

            
        trainer.fit(model=litmodule,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
        val_loss = best_val_checkpoint.best_model_score.item()

        # Delete the checkpoint directory
        os.system(f'rm -r ./checkpoints/{composition_timestamp}')

        final_epoch = trainer.current_epoch
        print(f"Finished after {final_epoch} epochs")

        if self.verbose:
            print(f'Validation loss for {composition}: {val_loss}')

        return val_loss


    def _fit_composition(self,composition_config,composition,train_dataset,val_dataset,tuning=False):


        if tuning:
            if self.verbose:
                print(f'Tuning the hyperparameters for the composition: {composition}')
            def objective(trial):
                parameters = {
                    # 'dis_loss_coeff_1': trial.suggest_float('dis_loss_coeff_1', 1e-9, 1e-1, log=True),
                    'dis_loss_coeff_2': trial.suggest_float('dis_loss_coeff_2', 1e-9, 1e-1, log=True),
                    'lr': trial.suggest_float('lr', 1e-4, 1.0, log=True),
                    'last_loss_coeff': trial.suggest_float('last_loss_coeff', 1e-3, 1e+3, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
                }
                val_loss = self._fit_composition_objective(composition_config,composition,train_dataset,val_dataset,True,parameters)
                return val_loss
            
            sampler = optuna.samplers.TPESampler(seed=self.config['seed'])
            study = optuna.create_study(sampler=sampler,direction='minimize')
            study.optimize(objective, n_trials=self.config['n_tune'])
            best_trial = study.best_trial
            best_hyperparameters = best_trial.params
            if self.verbose:
                print(f'Best hyperparameters: {best_hyperparameters}')
            for key, value in best_hyperparameters.items():
                composition_config[key] = value

        litmodule = LitSketchODE(composition_config)
        # if self.torch_models is not None:
        #     litmodule.model.fixed_infinite_properties = self.torch_models[property_index].fixed_infinite_properties

        composition_timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        if self.verbose:
            print(f'Fitting the model to the data using the composition: {composition}')

        val_loss = np.inf
        num_retries = 0
        while val_loss == np.inf or np.isnan(val_loss):
            if num_retries > 0:
                print(f"Retrying fitting the model with a different seed")
            # create callbacks
            best_val_checkpoint = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='best_val',
                dirpath=f'./checkpoints/{composition_timestamp}'
            )
            # added early stopping callback, if validation loss does not improve over 10 epochs -> terminate training.
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=20,
                verbose=False,
                mode='min'
            )
            callback_ls = [best_val_checkpoint, early_stop_callback]

            trainer_dict = {
                'deterministic': True,
                'devices': 1,
                'enable_model_summary': False,
                'enable_progress_bar': self.verbose,
                'accelerator': self.lightning_accelerator,
                'max_epochs': self.config['n_epochs'],
                'check_val_every_n_epoch': 5,
                'log_every_n_steps': 1,
                'callbacks': callback_ls
            }
            trainer = L.Trainer(**trainer_dict)

            torch_gen = torch.Generator()
            torch_gen.manual_seed(self.config['seed'])

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, generator=torch_gen)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        
                
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING) 
            logging.getLogger('lightning').setLevel(0)

            if num_retries > 0:
                new_config = copy.deepcopy(litmodule.config)
                new_config['seed'] += 1
                litmodule = LitSketchODE(new_config) 
            trainer.fit(model=litmodule,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
            val_loss = best_val_checkpoint.best_model_score.item()
            num_retries += 1

        best_model_path =  best_val_checkpoint.best_model_path
        best_model = LitSketchODE.load_from_checkpoint(checkpoint_path=best_model_path,config=litmodule.config)
        
        if self.verbose:
            print(f'Validation loss for {composition}: {val_loss}')

        if self.config['dtw']:
            if self.verbose:
                print("Tunning with MSE loss")
            best_model.refit = True
             # create callbacks
            best_val_checkpoint = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='best_val',
                dirpath=f'./checkpoints/{composition_timestamp}'
            )
            # added early stopping callback, if validation loss does not improve over 10 epochs -> terminate training.
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=20,
                verbose=False,
                mode='min'
            )
            callback_ls = [best_val_checkpoint, early_stop_callback]

            trainer_dict = {
                'deterministic': True,
                'devices': 1,
                'enable_model_summary': False,
                'enable_progress_bar': self.verbose,
                'accelerator': self.lightning_accelerator,
                'max_epochs': self.config['n_epochs'],
                'check_val_every_n_epoch': 5,
                'log_every_n_steps': 1,
                'callbacks': callback_ls
            }
            trainer = L.Trainer(**trainer_dict)

            torch_gen = torch.Generator()
            torch_gen.manual_seed(self.config['seed'])

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, generator=torch_gen)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        
            trainer.fit(model=best_model,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

            best_model_path =  best_val_checkpoint.best_model_path
            best_model = LitSketchODE.load_from_checkpoint(checkpoint_path=best_model_path,config=litmodule.config)


        # Delete the checkpoint directory
        os.system(f'rm -r ./checkpoints/{composition_timestamp}')

        final_epoch = trainer.current_epoch
        print(f"Finished after {final_epoch} epochs")
        
        # val_loss = trainer.callback_metrics['val_loss']

        
        if self.verbose:
            print(f'Validation loss for {composition}: {val_loss}')

        return val_loss, best_model
    

    def _get_default_opt_config(self):
        config = {
            'device': 'cpu',
            'n_epochs': 1000,
            'batch_size': 256,
            'lr': 1.0e-1,
            'weight_decay': 1e-4,
            'fit_single': True,
        }
        return config
    
    def _get_default_dt_config(self):
        config = {
            'max_depth': 3,
            'min_relative_gain_to_split': 1e-2,
            'min_samples_leaf':10,
            'relative_motif_cost': 1e-2,
        }
        return config