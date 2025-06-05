import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline
import numpy as np
from episode.infinite_motifs import *
from episode.soft_dtw import SoftDTW
import episode.utils as utils

MIN_TRANSITION_POINT_SEP = 0.01
# MIN_PROPERTY_VALUE = 1e-3
MIN_RELATIVE_DISTANCE_TO_LAST_FINITE_TRANSITION_POINT = 1e-1

def torch_inverse_softplus(x):
    # Above 20, the softplus is approximately equal to the input
    return torch.where(x < 20, torch.log(torch.exp(x) - 1), x)

class CubicModel(torch.nn.Module):
    def __init__(self, config, fixed_infinite_properties={}):
        super(CubicModel, self).__init__()

        self.infinite_motif_classes = {
            '++f': PPF(),
            '+-p': PMP(),
            '+-h': PMH(),
            '-+f': MPF(),
            '-+h': MPH(),
            '--f': MMF()
        }

        self.infinite_motif_single_classes = {
            '++f': PPFSingle(),
            '+-p': PMPSingle(),
            '+-h': PMHSingle(),
            '-+f': MPFSingle(),
            '-+h': MPHSingle(),
            '--f': MMFSingle(),
            '00h': ZZSingle(),
        }
        
        self.config = config
        
        self.composition = config['composition']
        self.n_basis_functions = config['n_basis_functions']
        self.n_features = config['n_features']
        self.seed = config['seed']
        self.n_coordinates = 2*len(self.composition)
        self.x0_included = config['x0_included']
        self.x0_index = config['x0_index']

        self.dis_loss_coeff_1 = config['dis_loss_coeff_1']
        self.dis_loss_coeff_2 = config['dis_loss_coeff_2']
        self.last_loss_coeff = config['last_loss_coeff']

        self.categorical_features_indices = config['categorical_features_indices']
        self.cat_n_unique_dict = config['cat_n_unique_dict']

        self.n_cat_features = len(self.categorical_features_indices)
        self.n_cont_features = self.n_features - self.n_cat_features

        self.fixed_infinite_properties = fixed_infinite_properties

        self.soft_constraint = config['soft_constraint']

        if config['device'] == 'gpu' or config['device'] == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if config['t_range'][1] is None:
            self.t_range = torch.Tensor([config['t_range'][0],np.inf]).to(self.device)
        else:
            self.t_range = torch.Tensor([config['t_range'][0],config['t_range'][1]]).to(self.device)
        
        torch.manual_seed(self.seed)
        
        if self.composition[-1][-1] != 'c':
            # it is an infinite composition
            self.composition_finite_part = self.composition[:-1]
            self.infinite_motif = self.composition[-1]
            self.infinite_composition = True
        else:
            # it is a finite composition
            self.composition_finite_part = self.composition
            self.infinite_motif = None
            self.infinite_composition = False

        self.scalers = {}

        # t-coordinates of transition points
        t_span = self.t_range[1] - self.t_range[0]
        avg_interval = t_span / len(self.composition_finite_part)
        avg_interval_per_feature = avg_interval / (self.n_features + 1)
        inverse_interval_per_feature = torch_inverse_softplus(avg_interval_per_feature).item()

        horizontal_weights = torch.randn(self.n_basis_functions, self.n_cont_features, len(self.composition_finite_part)) + inverse_interval_per_feature
        self.horizontal_weights = torch.nn.Parameter(horizontal_weights)
        self.cat_horizontal_weights = torch.nn.ParameterDict({
            str(i): torch.nn.Parameter(torch.randn(self.cat_n_unique_dict[i], len(self.composition_finite_part)) + inverse_interval_per_feature) for i in self.categorical_features_indices
        })
        self.horizontal_bias = torch.nn.Parameter(torch.randn(len(self.composition_finite_part)) + inverse_interval_per_feature)

        # x-coordinates of transition points
        self.vertical_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, self.n_cont_features, len(self.composition_finite_part)))
        self.cat_vertical_weights = torch.nn.ParameterDict({
            str(i): torch.nn.Parameter(torch.randn(self.cat_n_unique_dict[i], len(self.composition_finite_part))) for i in self.categorical_features_indices
        })
        self.vertical_bias = torch.nn.Parameter(torch.randn(len(self.composition_finite_part)))

        # TODO: This is only needed if there is no initial condition. Otherwise, it is redundant
        self.initial_condition_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, self.n_cont_features, 1))
        self.cat_initial_condition_weights = torch.nn.ParameterDict({
            str(i): torch.nn.Parameter(torch.randn(self.cat_n_unique_dict[i], 1)) for i in self.categorical_features_indices
        })
        self.initial_condition_bias = torch.nn.Parameter(torch.randn(1))

        # Special properties of the infinite motif
        if self.infinite_composition:
            num_properties = self.number_of_properties_for_infinite_motif()
            self.infinite_motif_properties_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, self.n_cont_features, num_properties))
            self.cat_infinite_motif_properties_weights = torch.nn.ParameterDict({
                str(i): torch.nn.Parameter(torch.randn(self.cat_n_unique_dict[i], num_properties)) for i in self.categorical_features_indices
            })
            self.infinite_motif_properties_bias = torch.nn.Parameter(torch.randn(num_properties))
            self.scalers['infinite_motif_properties'] = torch.nn.Parameter(torch.randn(num_properties))

        self.first_derivative_at_start_status = utils.get_first_derivative_at_start_status(self.composition)
        self.first_derivative_at_end_status = utils.get_first_derivative_at_end_status(self.composition)
        self.second_derivative_at_end_status = utils.get_second_derivative_at_end_status(self.composition)

        if self.first_derivative_at_start_status == "weights":
            self.first_derivative_at_start = torch.nn.Parameter(torch.randn(self.n_basis_functions, self.n_cont_features, 1))
            self.cat_first_derivative_at_start = torch.nn.ParameterDict({
                str(i): torch.nn.Parameter(torch.randn(self.cat_n_unique_dict[i], 1)) for i in self.categorical_features_indices
            })
            self.first_derivative_at_start_bias = torch.nn.Parameter(torch.randn(1))

        if self.first_derivative_at_end_status == "weights":
            self.first_derivative_at_end = torch.nn.Parameter(torch.randn(self.n_basis_functions, self.n_cont_features, 1))
            self.cat_first_derivative_at_end = torch.nn.ParameterDict({
                str(i): torch.nn.Parameter(torch.randn(self.cat_n_unique_dict[i], 1)) for i in self.categorical_features_indices
            })
            self.first_derivative_at_end_bias = torch.nn.Parameter(torch.randn(1))

        if self.second_derivative_at_end_status == "weights":
            self.second_derivative_at_end = torch.nn.Parameter(torch.randn(self.n_basis_functions, self.n_cont_features, 1))
            self.cat_second_derivative_at_end = torch.nn.ParameterDict({
                str(i): torch.nn.Parameter(torch.randn(self.cat_n_unique_dict[i], 1)) for i in self.categorical_features_indices
            })
            self.second_derivative_at_end_bias = torch.nn.Parameter(torch.randn(1))
        
        self.scalers = torch.nn.ParameterDict(self.scalers)

    def number_of_properties_for_infinite_motif(self):
        if not self.infinite_composition:
            return 0
        else:
            return self.infinite_motif_classes[self.infinite_motif].num_network_properties()
        
    def _first_derivative_at_start_from_weights(self,B,finite_coordinates):
        """
        Calculate the first derivative at the start from B and the finite coordinates

        Args:
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        finite_coordinates: input tensor of shape (batch_size, n_all_coordinates, 2)

        Returns:
        output tensor of shape (batch_size,)
        """
        motif_index = 0
        coordinate_1 = finite_coordinates[:,motif_index,:]
        coordinate_2 = finite_coordinates[:,motif_index+1,:]

        calculated_first_derivative_ratio = torch.sigmoid(self._calculate_gams(B,self.first_derivative_at_start,self.cat_first_derivative_at_start,self.first_derivative_at_start_bias,positive=False,sum=True)).flatten()
        slope_min, slope_max = self.get_first_derivative_range(0, coordinate_1, coordinate_2, "left")
        return slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
    
    def _first_derivative_at_end_from_weights(self,B,finite_coordinates=None):
        """
        Calculate the first derivative at the end from B

        Args:
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))

        Returns:
        output tensor of shape (batch_size,)
        """
        if self.infinite_composition:
            # This is only triggered if there is only one motif in the infinite composition. 
            # Otherwise, the first derivative at the end is calculated from the cubic coefficients or is set to zero.
            sign = 1 if self.composition[-1][0] == '+' else -1
            return sign * self._calculate_gams(B,self.first_derivative_at_end,self.cat_first_derivative_at_end,self.first_derivative_at_end_bias,positive=True,sum=True).flatten()
        else:
            # Only when there is more than one motif in the finite composition
            motif_index = len(self.composition) - 1
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            calculated_first_derivative_ratio = torch.sigmoid(self._calculate_gams(B,self.first_derivative_at_end,self.cat_first_derivative_at_end,self.first_derivative_at_end_bias,positive=False,sum=True)).flatten()
            slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "right")
            return slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)

    def _first_derivative_at_end_from_cubic(self, all_coefficients, finite_coordinates):
        """
        Calculate the first derivative at the end from the cubic coefficients

        Args:
        all_coefficients: input tensor of shape (batch_size, n_motifs, 4)
        finite_coordinates: input tensor of shape (batch_size, n_all_coordinates, 2)

        Returns:
        output tensor of shape (batch_size,)
        """
        last_transition_point = finite_coordinates[:,-1,:]
        last_first_derivative = 3*all_coefficients[:,-1,0] * last_transition_point[:,0]**2 + 2 * all_coefficients[:,-1,1] * last_transition_point[:,0] + all_coefficients[:,-1,2]
        return last_first_derivative
    
    def _second_derivative_at_end_from_weights(self,B):
        """
        Calculate the second derivative at the end from B

        Args:
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))

        Returns:
        output tensor of shape (batch_size,)
        """
        sign = 1 if self.composition[-1][1] == '+' else -1
        return sign * self._calculate_gams(B,self.second_derivative_at_end,self.cat_second_derivative_at_end,self.second_derivative_at_end_bias,positive=True,sum=True).flatten()
    
    def _second_derivative_at_end_from_cubic(self, all_coefficients, finite_coordinates):
        """
        Calculate the second derivative at the end from the cubic coefficients

        Args:
        all_coefficients: input tensor of shape (batch_size, n_motifs, 4)
        finite_coordinates: input tensor of shape (batch_size, n_all_coordinates, 2)

        Returns:
        output tensor of shape (batch_size,)
        """
        last_transition_point = finite_coordinates[:,-1,:]
        last_second_derivative = 6*all_coefficients[:,-1,0] * last_transition_point[:,0] + 2 * all_coefficients[:,-1,1]
        return last_second_derivative
    
    def _boundary_derivative(self,boundary,order,B,finite_coordinates,all_coefficients):
        """
        Calculate the boundary derivative

        Args:
        boundary: string, either 'start' or 'end'
        order: integer, either 1 or 2
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        finite_coordinates: input tensor of shape (batch_size, n_all_coordinates, 2)
        all_coefficients: input tensor of shape (batch_size, n_motifs, 4)

        Returns:
        output tensor of shape (batch_size,)
        """
        batch_size = B[0].shape[0]

        if boundary == 'start':
            if order == 1:
                if self.first_derivative_at_start_status == "weights":
                    return self._first_derivative_at_start_from_weights(B,finite_coordinates)
                elif self.first_derivative_at_start_status == "none":
                    return None
            elif order == 2:
                raise ValueError('Second derivative at the start is not implemented')
        elif boundary == 'end':
            if order == 1:
                if self.first_derivative_at_end_status == "weights":
                    return self._first_derivative_at_end_from_weights(B)
                elif self.first_derivative_at_end_status == "cubic":
                    return self._first_derivative_at_end_from_cubic(all_coefficients, finite_coordinates)
                elif self.first_derivative_at_end_status == "zero":
                    return torch.zeros(batch_size).to(self.device)
            elif order == 2:
                if self.second_derivative_at_end_status == "weights":
                    return self._second_derivative_at_end_from_weights(B)
                elif self.second_derivative_at_end_status == "cubic":
                    return self._second_derivative_at_end_from_cubic(all_coefficients, finite_coordinates)
                elif self.second_derivative_at_end_status == "zero":
                    return torch.zeros(batch_size).to(self.device)
                
    def _infinite_properties_from_weights(self,B):
        """
        Calculate the infinite properties from B

        Args:
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))

        Returns:
        output tensor of shape (batch_size, n_properties)
        """
        return self._calculate_gams(B,self.infinite_motif_properties_weights,self.cat_infinite_motif_properties_weights,self.infinite_motif_properties_bias,positive=True,sum=True) * torch.nn.functional.softplus(self.scalers['infinite_motif_properties'])
    
    def extract_boundary_derivative(self,boundary,order,B,X0=None):
        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(B, X0=X0)
        return self._boundary_derivative(boundary,order,B,finite_coordinates,all_coefficients)

           
    def extract_first_derivative_at_start(self,B,X0=None):
        """
        Extract the first derivative at the start

        Args:
        B: a pair of input tensor of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        X0 (optional): input tensor of shape (batch_size, 1)

        Returns:
        output tensor of shape (batch_size,)
        """
        self.extract_boundary_derivative('start',1,B,X0=X0)
        
    def extract_first_derivative_at_end(self,B,X0=None):
        """
        Extract the first derivative at the end

        Args:
        B: a pair input tensor of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        X0 (optional): input tensor of shape (batch_size, 1)

        Returns:
        output tensor of shape (batch_size,)
        """
        self.extract_boundary_derivative('end',1,B,X0=X0)

    def extract_second_derivative_at_end(self,B,X0=None):
        """
        Extract the second derivative at the end

        Args:
        B: pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        X0 (optional): input tensor of shape (batch_size, 1)

        Returns:
        output tensor of shape (batch_size,)
        """
        self.extract_boundary_derivative('end',2,B,X0=X0)
       
    
    def extract_properties_infinite_motif(self, B, X0=None):
        """
        
        Args:
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        X0 (optional): input tensor of shape (batch_size, 1)
        """
        batch_size = B[0].shape[0]
        if not self.infinite_composition:
            return torch.zeros(batch_size, self.number_of_properties_for_infinite_motif()).to(self.device)

        properties = self._infinite_properties_from_weights(B)
        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(B, X0=X0)
        last_first_derivative = self._boundary_derivative('end',1,B,finite_coordinates,all_coefficients)
        last_second_derivative = self._boundary_derivative('end',2,B,finite_coordinates,all_coefficients)
            
        if self.fixed_infinite_properties is not None:
            raise NotImplementedError('Fixed infinite properties are not implemented yet')
            # for index, property_function in self.fixed_infinite_properties.items():
            #     properties[:,index] = property_function(X, x0, y0, last_first_derivative, last_second_derivative)

        return self._extract_print_properties_infinite_motif(properties, finite_coordinates, last_first_derivative, last_second_derivative)
    
    def _extract_print_properties_infinite_motif(self, raw_properties_from_weights, finite_coordinates, last_first_derivative, last_second_derivative):
        """
        Extract the properties of the infinite motif

        Args:
        finite_coordinates: input tensor of shape (batch_size, n_all_coordinates, 2)
        last_first_derivative: input tensor of shape (batch_size,)
        last_second_derivative: input tensor of shape (batch_size,)

        Returns:
        output tensor of shape (batch_size, n_properties)
        """
        last_transition_point = finite_coordinates[:,-1,:]
        x0 = last_transition_point[:,[0]]
        y0 = last_transition_point[:,[1]]
        motif_class = self.infinite_motif_classes[self.infinite_motif]
        return motif_class.extract_properties_from_network(raw_properties_from_weights, x0, y0, last_first_derivative.unsqueeze(-1), last_second_derivative.unsqueeze(-1))
    

    def _divide_B_cat_into_blocks(self, B_cat):
        """
        
        Args:
        B_cat: input tensor of shape (batch_size, sum(cat_n_unique_dict))

        Returns:
        dictionary of tensors, each of shape (batch_size, cat_n_unique_dict)
        """

        cat_block_dict = {}

        start_index = 0
        for cat_feature_index in self.categorical_features_indices:
            end_index = start_index + self.cat_n_unique_dict[cat_feature_index]
            cat_block = B_cat[:,start_index:end_index]
            cat_block_dict[str(cat_feature_index)] = cat_block
            start_index = end_index
    
        return cat_block_dict

    def _calculate_cat_shape_functions(self, B_cat, cat_weights, positive=True):
        """
        
        Args:
        B_cat: input tensor of shape (batch_size, sum(cat_n_unique_dict))
        cat_weights: a dictionary of tensors, each of shape (cat_n_unique_dict, n_properties)
        positive: boolean, whether the output should be positive or not

        Returns:
        output tensor of shape (batch_size, n_properties, n_cat_features)
        """

        # Divide B_cat into blocks for each variable
        cat_block_dict = self._divide_B_cat_into_blocks(B_cat)

        # Calculate the shape functions for each variable
        cat_shape_functions = {}
        for cat_feature_index in self.categorical_features_indices:
            cat_block = cat_block_dict[str(cat_feature_index)]
            cat_weight = cat_weights[str(cat_feature_index)]
            if positive:
                cat_shape_functions[str(cat_feature_index)] = torch.nn.functional.softplus(cat_block @ cat_weight) # shape (batch_size, n_properties)
            else:
                cat_shape_functions[str(cat_feature_index)] = cat_block @ cat_weight # shape (batch_size, n_properties)
        
        # Stack the shape functions
        return torch.stack([cat_shape_functions[str(cat_feature_index)] for cat_feature_index in self.categorical_features_indices], dim=2) # shape (batch_size, n_properties, n_cat_features)
        
    def _calculate_gams(self, B, weights, cat_weights, bias, positive=True, sum=False):
        """
        Calculate the GAMS of the model

        Args:
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        weights: input tensor of shape (n_basis_functions, n_cont_features, n_properties)
        cat_weights: a dictionary of tensors, each of shape (cat_n_unique_dict, n_properties)
        bias: input tensor of shape (n_properties)

        Returns:
        output tensor of shape (batch_size, n_properties, n_features + 1) if sum is False else (batch_size, n_properties)
        """
        B, B_cat = B
        if positive:
            shape_functions = torch.nn.functional.softplus(torch.einsum('dmb,bmp->dpm', B, weights))
            shape_bias = torch.tile(torch.nn.functional.softplus(bias), (B.shape[0],1)).unsqueeze(-1)
        else:
            shape_functions = torch.einsum('dmb,bmp->dpm', B, weights)
            shape_bias = torch.tile(bias, (B.shape[0],1)).unsqueeze(-1)
        
        if len(self.categorical_features_indices) > 0:
            cat_shape_functions = self._calculate_cat_shape_functions(B_cat, cat_weights, positive=positive)
            shape_functions_with_bias = torch.concat([shape_functions, cat_shape_functions, shape_bias], dim=2) # shape (batch_size, n_properties, n_features + 1)
        else:
            shape_functions_with_bias = torch.concat([shape_functions, shape_bias], dim=2) # shape (batch_size, n_properties, n_features + 1)
        if sum:
            return torch.sum(shape_functions_with_bias, dim=2)
        else:
            return shape_functions_with_bias
        

    def _from_feature_index_to_cont_feature_index(self, feature_index):
        """
        Convert a feature index to a continuous feature index

        Args:
        feature_index: integer, index of the feature

        Returns:
        output integer
        """
        if feature_index in self.categorical_features_indices:
            return None
        cont_feature_index = 0
        for i in range(feature_index):
            if i not in self.categorical_features_indices:
                cont_feature_index += 1
        
        return cont_feature_index
    
    def _calculate_shape_functions_for_single_feature(self, b, feature_index, cont_weights, cat_weights, positive=True):
        """
        
        Args:
        b: input tensor of shape (batch_size, n_basis_functions) or (batch_size, cat_n_unique) if categorical
        feature_index: integer, index of the feature from [n_features]
        cont_weights: input tensor of shape (n_basis_functions, n_cont_features, n_properties) or (cat_n_unique, n_properties) if categorical

        Returns:
        output tensor of shape (batch_size, n_properties)
        """
        if feature_index in self.categorical_features_indices:
            raw_shape_functions = b @ cat_weights[str(feature_index)] # shape (batch_size, n_properties)
        else:
            cont_feature_index = self._from_feature_index_to_cont_feature_index(feature_index)
            raw_shape_functions = b @ cont_weights[:,cont_feature_index,:] # shape (batch_size, n_properties)
        
        if positive:
            return torch.nn.functional.softplus(raw_shape_functions)
        else:
            return raw_shape_functions
        

    def extract_shape_function_of_transition_point(self, ind, coordinate, feature_index, b, X0=None):
        """
        Extract the shape function of a transition point

        Args:
        ind: integer, index of the transition point
        coordinate: string, either 't' or 'x'
        feature_index: integer, index of the feature from [n_features]
        b: input tensor of shape (batch_size, n_basis_functions) or (batch_size, cat_n_unique) if categorical
        X0 (optional): input tensor of shape (batch_size, 1)

        Returns:
        output tensor of shape (batch_size,)
        """

        if coordinate == 't':
            if not self.infinite_composition:
                if ind == len(self.composition_finite_part): # the last transition point
                    shape_function = torch.zeros(b.shape[0]).to(self.device)
                    return shape_function
            raw_shape_functions = self._calculate_shape_functions_for_single_feature(b, feature_index, self.horizontal_weights, self.cat_horizontal_weights, positive=True)
            shape_function = torch.sum(raw_shape_functions[:,:ind], dim=1)
            return shape_function
        elif coordinate == 'x':
            raw_shape_functions = self._calculate_shape_functions_for_single_feature(b, feature_index, self.vertical_weights, self.cat_vertical_weights, positive=True)
            if self.x0_included:
                if isinstance(self.x0_index, int):
                    if feature_index == self.x0_index:
                        initial_condition_shape_function = X0
                    else:
                        initial_condition_shape_function = torch.zeros(b.shape[0], 1).to(self.device)
                elif isinstance(self.x0_index, float):
                    initial_condition_shape_function = torch.zeros(b.shape[0], 1).to(self.device)
                else:
                    raise ValueError('Invalid x0_index')
            else:
                initial_condition_shape_function = self._calculate_shape_functions_for_single_feature(b, feature_index, self.initial_condition_weights, self.cat_initial_condition_weights, positive=False)
            
            cumulative_vertical_shape_functions = initial_condition_shape_function # shape (batch_size, 1)

            for j in range(1,ind+1):
                sign = 1 if self.composition_finite_part[j-1][0] == '+' else -1
                cumulative_vertical_shape_functions += sign * raw_shape_functions[:,[j-1]]

            return cumulative_vertical_shape_functions.flatten()
        
    def extract_bias_of_transition_point(self, ind, coordinate):
        """
        Extract the bias of a transition point

        Args:
        ind: integer, index of the transition point
        coordinate: string, either 't' or 'x'

        Returns:
        output tensor of shape (batch_size,)
        """
        if coordinate == 't':
            if not self.infinite_composition:
                if ind == len(self.composition_finite_part): # the last transition point
                    return self.t_range[1]
            raw_bias = torch.nn.functional.softplus(self.horizontal_bias)
            return torch.sum(raw_bias[:ind]) + self.t_range[0] + MIN_TRANSITION_POINT_SEP * ind
        elif coordinate == 'x':
            raw_bias = torch.nn.functional.softplus(self.vertical_bias)

            if self.x0_included:
                if isinstance(self.x0_index, int):
                    initial_condition_bias = torch.tensor([0.0]).to(self.device)
                elif isinstance(self.x0_index, float):
                    initial_condition_bias = torch.ones(1).to(self.device) * self.x0_index
            else:
                initial_condition_bias = self.initial_condition_bias
            
            cumulative_vertical_bias = initial_condition_bias

            for j in range(1,ind+1):
                sign = 1 if self.composition_finite_part[j-1][0] == '+' else -1
                cumulative_vertical_bias += sign * raw_bias[j-1]
            return cumulative_vertical_bias
        
    def extract_shape_function_of_derivative(self, boundary, order, feature_index, b, X0=None):
        """
        Extract the shape function of a derivative

        Args:
        boundary: string, either 'start' or 'end'
        order: integer, either 1 or 2
        feature_index: integer, index of the feature
        b: input tensor of shape (batch_size, n_basis_functions) or (batch_size, cat_n_unique) if categorical
        X0 (optional): input tensor of shape (batch_size, 1)

        Returns:
        output tensor of shape (batch_size,)
        """
        if boundary == 'start':
            if order == 1:
                if self.first_derivative_at_start_status == "weights":
                    return self._calculate_shape_functions_for_single_feature(b, feature_index, self.first_derivative_at_start, self.cat_first_derivative_at_start, positive=False)[:,0]
                else:
                    raise ValueError('First derivative at the start is not implemented as a trainable GAM')
            else:
                raise ValueError('Second derivative at the start is not implemented')
        elif boundary == 'end':
            if order == 1:
                if self.first_derivative_at_end_status == "weights":
                    if self.infinite_composition:
                        sign = 1 if self.composition[-1][0] == '+' else -1
                        return sign * self._calculate_shape_functions_for_single_feature(b, feature_index, self.first_derivative_at_end, self.cat_first_derivative_at_end, positive=True)[:,0]
                    else:
                        # This would need to be passed by a sigmoid later
                        return self._calculate_shape_functions_for_single_feature(b, feature_index, self.first_derivative_at_end, self.cat_first_derivative_at_end, positive=True)[:,0]
                else:
                    raise ValueError('First derivative at the end is not implemented as a trainable GAM')
            elif order == 2:
                if self.second_derivative_at_end_status == "weights":
                    sign = 1 if self.composition[-1][1] == '+' else -1
                    return sign * self._calculate_shape_functions_for_single_feature(b, feature_index, self.second_derivative_at_end, self.cat_second_derivative_at_end, positive=True)[:,0]
                else:
                    raise ValueError('Second derivative at the end is not implemented as a trainable GAM')
        
    def extract_bias_of_derivative(self, boundary, order):
        """
        Extract the bias of a derivative

        Args:
        boundary: string, either 'start' or 'end'
        order: integer, either 1 or 2

        Returns:
        output tensor of shape (1,)
        """
        if boundary == 'start':
            if order == 1:
                if self.first_derivative_at_start_status == "weights":
                    return self.first_derivative_at_start_bias
                else:
                    return torch.zeros(1).to(self.device)
            else:
                raise ValueError('Second derivative at the start is not implemented')
        elif boundary == 'end':
            if order == 1:
                if self.first_derivative_at_end_status == "weights":
                    sign = 1 if self.composition[-1][0] == '+' else -1
                    return sign * torch.nn.functional.softplus(self.first_derivative_at_end_bias)
                else:
                    return torch.zeros(1).to(self.device)
            elif order == 2:
                if self.second_derivative_at_end_status == "weights":
                    sign = 1 if self.composition[-1][1] == '+' else -1
                    return sign * torch.nn.functional.softplus(self.second_derivative_at_end_bias)
                else:
                    return torch.zeros(1).to(self.device)

    
    def extract_shape_function_of_infinite_property(self, property_index, feature_index, b, X0=None):
        """
        Extract the shape function of an infinite property

        Args:
        property_index: integer, index of the property
        feature_index: integer, index of the feature
        b: input tensor of shape (batch_size, n_basis_functions) or (batch_size, cat_n_unique) if categorical
        X0 (optional): input tensor of shape (batch_size, 1)

        Returns:
        output tensor of shape (batch_size,)
        """
        raw_shape_functions = self._calculate_shape_functions_for_single_feature(b, feature_index, self.infinite_motif_properties_weights, self.cat_infinite_motif_properties_weights, positive=True) * torch.nn.functional.softplus(self.scalers['infinite_motif_properties'])
        last_transition_point_index = len(self.composition_finite_part)
        x0_shape_functions = self.extract_shape_function_of_transition_point(last_transition_point_index, 't', feature_index, b, X0=X0).reshape(-1,1)
        y0_shape_functions = self.extract_shape_function_of_transition_point(last_transition_point_index, 'x', feature_index, b, X0=X0).reshape(-1,1)
        motif_class = self.infinite_motif_classes[self.infinite_motif]
        return motif_class.get_property_shapes(raw_shape_functions, x0_shape_functions, y0_shape_functions)[:,property_index]
    
    def extract_bias_of_infinite_property(self, property_index):
        """
        Extract the bias of an infinite property

        Args:
        property_index: integer, index of the property

        Returns:
        output tensor of shape (1,)
        """
        raw_bias = torch.nn.functional.softplus(self.infinite_motif_properties_bias) * torch.nn.functional.softplus(self.scalers['infinite_motif_properties'])
        raw_bias = raw_bias.reshape(1,-1)
        last_transition_point_index = len(self.composition_finite_part)
        x0_bias = self.extract_bias_of_transition_point(last_transition_point_index, 't').reshape(1,1)
        y0_bias = self.extract_bias_of_transition_point(last_transition_point_index, 'x').reshape(1,1)
        motif_class = self.infinite_motif_classes[self.infinite_motif]
        return motif_class.get_property_shapes(raw_bias, x0_bias, y0_bias)[0,property_index]


    
    def extract_coordinates_finite_composition(self, B, X0=None):
        """
        Args:
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        """

        calculated_values_horizontal = self._calculate_gams(B,self.horizontal_weights,self.cat_horizontal_weights,self.horizontal_bias)
        calculated_values_vertical = self._calculate_gams(B,self.vertical_weights,self.cat_vertical_weights,self.vertical_bias)

        # For the blood donation dataset
        # if X0 is not None:
        #     calculated_initial_condition = - self._calculate_gams(B,self.initial_condition_weights,self.initial_condition_bias, positive=True)
        #     calculated_initial_condition[:,0,-1] += X0.flatten()
        # else:
        #     calculated_initial_condition = self._calculate_gams(B,self.initial_condition_weights,self.initial_condition_bias, positive=False)
        
        batch_size = B[0].shape[0]
        # For rebuttal
        if X0 is not None:
            if isinstance(self.x0_index, int):
                calculated_initial_condition = torch.zeros(batch_size, 1, self.n_features + 1).to(self.device)
                calculated_initial_condition[:,0,self.x0_index] = X0.flatten()
            elif isinstance(self.x0_index, float):
                calculated_initial_condition = torch.zeros(batch_size, 1, self.n_features + 1).to(self.device)
                calculated_initial_condition[:,0,-1] = self.x0_index # We set the bias to the x0_index value
            else:
                raise ValueError('Invalid x0_index')
        else:
            calculated_initial_condition = self._calculate_gams(B,self.initial_condition_weights,self.cat_initial_condition_weights,self.initial_condition_bias, positive=False)

        all_coordinate_values = torch.zeros(batch_size, len(self.composition_finite_part)+1,2).to(self.device)
        
        cumulative_horizontal_shape_functions = torch.zeros(batch_size, 1, self.n_features + 1).to(self.device) 
        cumulative_horizontal_shape_functions[:,0,-1] = self.t_range[0]

        cumulative_vertical_shape_functions = calculated_initial_condition # shape (batch_size, 1, n_features + 1)

        # j = 0
        all_coordinate_values[:,0,0] = torch.sum(cumulative_horizontal_shape_functions, dim=2)[:,0]
        all_coordinate_values[:,0,1] = torch.sum(cumulative_vertical_shape_functions, dim=2)[:,0]

        for j in range(1, len(self.composition_finite_part)+1):
            sign = 1 if self.composition_finite_part[j-1][0] == '+' else -1
            cumulative_horizontal_shape_functions += calculated_values_horizontal[:,[j-1],:]
            cumulative_vertical_shape_functions += sign * calculated_values_vertical[:,[j-1],:]

            all_coordinate_values[:,j,0] = torch.sum(cumulative_horizontal_shape_functions, dim=2)[:,0] + MIN_TRANSITION_POINT_SEP * j
            all_coordinate_values[:,j,1] = torch.sum(cumulative_vertical_shape_functions, dim=2)[:,0] #TODO: Maybe add minimum separation as in model_numpy.py

        if not self.infinite_composition:
            all_coordinate_values[:,-1,0] = torch.maximum(all_coordinate_values[:,-2,0] + MIN_TRANSITION_POINT_SEP, self.t_range[1])
        return all_coordinate_values

    def type_of_transition_point(self, ind):
        composition = self.composition
        if ind == 0:
            return 'start'
        elif ind == len(composition):
            return 'end'
        else:
            if (composition[ind-1][:2] == "++") and (composition[ind][:2] == "+-"):
                return 'inflection'
            elif (composition[ind-1][:2] == "+-") and (composition[ind][:2] == "++"):
                return 'inflection'
            elif (composition[ind-1][:2] == "+-") and (composition[ind][:2] == "--"):
                return 'max'
            elif (composition[ind-1][:2] == "-+") and (composition[ind][:2] == "++"):
                return 'min'
            elif (composition[ind-1][:2] == "-+") and (composition[ind][:2] == "--"):
                return 'inflection'
            elif (composition[ind-1][:2] == "--") and (composition[ind][:2] == "-+"):
                return 'inflection'
            else:
                raise ValueError('Unknown transition point type')
            
    def get_first_derivative_range_coefficients(self, motif_index, which_point):
        motif = self.composition[motif_index]
                                 
        if which_point == 'left':
            if motif == '++c':
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return (0,1)
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return (0,1)
            elif motif == "+-c":
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return (1,2)
                elif self.type_of_transition_point(motif_index+1) == 'max':
                    return (1.5,3)
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return (1,3)
            elif motif == "-+c":
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return (1,2)
                elif self.type_of_transition_point(motif_index+1) == 'min':
                    return (1.5,3)
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return (1,3)
            elif motif == "--c":
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return (0,1)
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return (0,1)
        
        elif which_point == 'right':
            if motif == '++c':
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return (1,3)
                elif self.type_of_transition_point(motif_index) == 'min':
                    return (1.5,3)
            elif motif == "+-c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return (0,1)
            elif motif == "-+c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return (0,1)
            elif motif == "--c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return (1,3)
                elif self.type_of_transition_point(motif_index) == 'max':
                    return (1.5,3)

    def get_first_derivative_range(self, motif_index, point1, point2, which_point):
        slope = (point2[:,1] - point1[:,1])/(point2[:,0] - point1[:,0])

        slope = slope.to(self.device)

        coefficients = self.get_first_derivative_range_coefficients(motif_index, which_point)

        return coefficients[0] * slope, coefficients[1] * slope

    def _create_row(self,coordinate, order):
        batch_size = coordinate.shape[0]
        if order == 0:
            return torch.stack([coordinate[:,0]**3, coordinate[:,0]**2, coordinate[:,0], torch.ones(batch_size).to(self.device)], dim=1)
        elif order == 1:
            return torch.cat([3*coordinate[:,[0]]**2, 2*coordinate[:,[0]], torch.ones(batch_size, 1).to(self.device), torch.zeros(batch_size, 1).to(self.device)], dim=1)
        elif order == 2:
            return torch.cat([6*coordinate[:,[0]], 2*torch.ones(batch_size, 1).to(self.device), torch.zeros(batch_size, 1).to(self.device), torch.zeros(batch_size, 1).to(self.device)], dim=1)
                
    def get_coefficients_and_coordinates_finite_composition(self, B, X0=None):

        finite_coordinates = self.extract_coordinates_finite_composition(B, X0=X0) # shape (batch_size, n_all_coordinates, 2)

        batch_size = B[0].shape[0]
        
        b = torch.zeros(batch_size, 3).to(self.device)
        coefficients_list = []

        for motif_index, motif in enumerate(self.composition_finite_part):
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            A_row_0 = self._create_row(coordinate_1, 0)
            A_row_1 = self._create_row(coordinate_2, 0)
            b_0 = coordinate_1[:,1]
            b_1 = coordinate_2[:,1]

            type_1 = self.type_of_transition_point(motif_index)
            type_2 = self.type_of_transition_point(motif_index+1)
            if type_1 == 'max' or type_1 == 'min':
                A_row_2 = self._create_row(coordinate_1, 1)
                b_2 = torch.zeros(batch_size).to(self.device)
            elif type_1 == 'inflection':
                A_row_2 = self._create_row(coordinate_1, 2)
                b_2 = torch.zeros(batch_size).to(self.device)
            elif type_1 == 'start' and self.first_derivative_at_start_status == "weights":
                A_row_2 = self._create_row(coordinate_1, 1)
                b_2 = self._first_derivative_at_start_from_weights(B,finite_coordinates)
                if type_2 == 'end':
                    # in that case we reduce the cubic to a quadratic #TODO: Why? why can't we just use the cubic?
                    A_row_3 = torch.cat([torch.ones(batch_size,1), torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)], dim=1).to(self.device)
                    b_3 = torch.zeros(batch_size).to(self.device)
            if type_2 == 'max' or type_2 == 'min':
                A_row_3 = self._create_row(coordinate_2, 1)
                b_3 = torch.zeros(batch_size).to(self.device)
            elif type_2 == 'inflection':
                A_row_3 = self._create_row(coordinate_2, 2)
                b_3 = torch.zeros(batch_size).to(self.device)
            elif (type_2 == 'end' and self.first_derivative_at_end_status == 'weights') and type_1 != 'start':
                # Ths is only possible if we have a finite composition
                A_row_3 = self._create_row(coordinate_2, 1)
                b_3 = self._first_derivative_at_end_from_weights(B, finite_coordinates)

            A = torch.stack([A_row_0, A_row_1, A_row_2, A_row_3], dim=1)
            b = torch.stack([b_0, b_1, b_2, b_3], dim=1)

            # Calculate the determinants of A
            determinants = torch.linalg.det(A)
            singular_indices = torch.abs(determinants) < 1e-6

            coefficients = torch.zeros(batch_size, 4).to(self.device)

            # For singular matrices, just connect with a line
            slope = (coordinate_2[singular_indices,1] - coordinate_1[singular_indices,1])/(coordinate_2[singular_indices,0]-coordinate_1[singular_indices,0])
            line_b = coordinate_1[singular_indices,1] - slope * coordinate_1[singular_indices,0]
            singular_coefficients = torch.stack([torch.zeros_like(line_b).to(self.device),torch.zeros_like(line_b).to(self.device),slope,line_b],dim=1)

            try:
                non_singular_coefficients = torch.linalg.solve(A[~singular_indices,:,:], b[~singular_indices,:])
            except:
                print(determinants)
                print(singular_indices)
                print(A[~singular_indices,:,:])
                # just connect with a line
                slope = (coordinate_2[~singular_indices,1] - coordinate_1[~singular_indices,1])/(coordinate_2[~singular_indices,0]-coordinate_1[~singular_indices,0])
                line_b = coordinate_1[~singular_indices,1] - slope * coordinate_1[~singular_indices,0]
                non_singular_coefficients = torch.stack([torch.zeros_like(line_b).to(self.device),torch.zeros_like(line_b).to(self.device),slope,line_b],dim=1)


            coefficients[singular_indices,:] = singular_coefficients
            coefficients[~singular_indices,:] = non_singular_coefficients


            # if torch.any(torch.abs(torch.linalg.det(A)) < 1e-9):
            #     # just connect with a line
            #     slope = (coordinate_2[:,1] - coordinate_1[:,1])/(coordinate_2[:,0]-coordinate_1[:,0])
            #     b = coordinate_1[:,1] - slope * coordinate_1[:,0]
            #     coefficients = torch.stack([torch.zeros_like(b).to(self.device),torch.zeros_like(b).to(self.device),slope,b],dim=1)
            # else:
            #     coefficients = torch.linalg.solve(A, b)
            
            coefficients_list.append(coefficients)
        if len(coefficients_list) == 0:
            all_coefficients = torch.zeros(batch_size, 0, 4).to(self.device)
        else:
            all_coefficients = torch.stack(coefficients_list, dim=1) # shape (batch_size, n_motifs, 4)
        return all_coefficients, finite_coordinates

    def evaluate_piece(self,finite_motif_coefficients, infinite_motif_properties, motif_index, T, last_transition_point=None,last_1st_derivative=None,last_2nd_derivative=None):
        if self.composition[motif_index][2] == 'c':
            # finite motif
            a = finite_motif_coefficients[:,[motif_index],0].repeat(1,T.shape[1])
            b = finite_motif_coefficients[:,[motif_index],1].repeat(1,T.shape[1])
            c = finite_motif_coefficients[:,[motif_index],2].repeat(1,T.shape[1])
            d = finite_motif_coefficients[:,[motif_index],3].repeat(1,T.shape[1])
            return a*T**3 + b*T**2 + c*T + d
        else:
            x0 = last_transition_point[:,[0]]
            y0 = last_transition_point[:,[1]]
            y1 = last_1st_derivative
            y2 = last_2nd_derivative

            motif_class = self.infinite_motif_classes[self.infinite_motif]
            T_to_use = torch.where(T < x0, x0, T) # make sure we don't evaluate the infinite motif before the last transition point - this might cause errors
            return motif_class.evaluate_from_network(T_to_use,infinite_motif_properties, x0, y0, y1, y2)
            
    def forward(self,X, B, T, X0=None):
        """
        Forward pass of the model

        Args:
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        T: input tensor of shape (batch_size, n_time_points)
        X0 (optional): input tensor of shape (batch_size, 1)
        """

        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(B, X0=X0)

        return self._forward(all_coefficients, finite_coordinates, B, T, X0=X0)
    
    def _forward(self, all_coefficients, finite_coordinates, B, T, X0=None):
        """
        Forward pass of the model

        Args:
        X: input tensor of shape (batch_size, 1)
        B: a pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        T: input tensor of shape (batch_size, n_time_points)
        """

        knots = finite_coordinates[:,:,0]
        last_transition_point = finite_coordinates[:,-1,:]
        batch_size = T.shape[0]
        
        if self.infinite_composition:
            # add infinite knots
            knots = torch.cat([knots, torch.from_numpy(np.array([np.inf])).reshape(1,1).repeat(batch_size,1).to(self.device)],dim=1)
            properties = self._infinite_properties_from_weights(B)
        else:
            properties = None

        Y_pred = torch.zeros(batch_size, T.shape[1]).to(self.device)

        for i in range(len(self.composition)):
            if self.composition[i][2] != 'c':
                last_first_derivative = self._boundary_derivative('end',1,B,finite_coordinates,all_coefficients).unsqueeze(-1)
                last_second_derivative = self._boundary_derivative('end',2,B,finite_coordinates,all_coefficients).unsqueeze(-1) 
            else:
                last_first_derivative = None
                last_second_derivative = None

            # TODO: Make fixing work
            # if self.fixed_infinite_properties is not None:
            #     x0 = last_transition_point[:,[0]]
            #     y0 = last_transition_point[:,[1]]
            #     y1 = last_first_derivative
            #     y2 = last_second_derivative
     
                # for index, property_function in self.fixed_infinite_properties.items():
                #     properties[:,index] = property_function(X, x0, y0, y1, y2)
            
            evaluated_piece = self.evaluate_piece(all_coefficients,properties,i,T,last_transition_point,last_first_derivative,last_second_derivative)

            Y_pred += torch.where((knots[:,[i]].repeat(1,T.shape[1]) <= T) & (T < knots[:,[i+1]].repeat(1,T.shape[1])),evaluated_piece,0)
        
        if not self.infinite_composition:
            # Due the sharp inequalities earlier, we need to add the last piece separately
            a = all_coefficients[:,[-1],0].repeat(1,T.shape[1])
            b = all_coefficients[:,[-1],1].repeat(1,T.shape[1])
            c = all_coefficients[:,[-1],2].repeat(1,T.shape[1])
            d = all_coefficients[:,[-1],3].repeat(1,T.shape[1])
            Y_pred += torch.where(T == knots[:,[-1]].repeat(1,T.shape[1]),a*T**3 + b*T**2 + c*T + d,0)
            # possibly add values beyond last t based on some condition, you can use first or second derivite information
        
        return Y_pred
    
    def loss_discontinuity_of_derivatives(self, coefficients, finite_coordinates):


        def derivative_difference_loss():

            global_first_derivative_discontinuity = 0
            global_second_derivative_discontinuity = 0
            
            for i in range(len(self.composition_finite_part)-1):
                if self.type_of_transition_point(i+1) == 'min' or self.type_of_transition_point(i+1) == 'max':
                    pass
                else:
                    first_derivatives_left = 3*coefficients[:,i,0] * finite_coordinates[:,i+1,0]**2 + 2 * coefficients[:,i,1] * finite_coordinates[:,i+1,0] + coefficients[:,i,2]
                    first_derivatives_right = 3*coefficients[:,i+1,0] * finite_coordinates[:,i+1,0]**2 + 2 * coefficients[:,i+1,1] * finite_coordinates[:,i+1,0] + coefficients[:,i+1,2]

                    first_derivative_discontinuity = first_derivatives_left - first_derivatives_right

                    first_norm = torch.max(torch.abs(first_derivatives_left), torch.abs(first_derivatives_right))
                    mask = first_norm > 1e-3

                    first_derivative_discontinuity[mask] =  first_derivative_discontinuity[mask]/first_norm[mask]
                    first_derivative_discontinuity[~mask] = first_derivative_discontinuity[~mask]

                    global_first_derivative_discontinuity += torch.mean(first_derivative_discontinuity ** 2)
                if self.type_of_transition_point(i+1) == 'inflection':
                    pass
                else:
                    second_derivatives_left = 6*coefficients[:,i,0] * finite_coordinates[:,i+1,0] + 2 * coefficients[:,i,1]
                    second_derivatives_right = 6*coefficients[:,i+1,0] * finite_coordinates[:,i+1,0] + 2 * coefficients[:,i+1,1]

                    second_derivative_discontinuity = second_derivatives_left - second_derivatives_right

                    second_norm = torch.max(torch.abs(second_derivatives_left), torch.abs(second_derivatives_right))
                    mask = second_norm > 1e-3

                    second_derivative_discontinuity[mask] =  second_derivative_discontinuity[mask]/second_norm[mask]
                    second_derivative_discontinuity[~mask] = second_derivative_discontinuity[~mask]*1e3

                    global_second_derivative_discontinuity += torch.mean(second_derivative_discontinuity ** 2)

            # calculate the loss
            return global_first_derivative_discontinuity + global_second_derivative_discontinuity

        def last_derivative_loss():
             # calculate the first derivative at the transition points
            first_derivative_last = 3*coefficients[:,-1,0] * finite_coordinates[:,-1,0]**2 + 2 * coefficients[:,-1,1] * finite_coordinates[:,-1,0] + coefficients[:,-1,2]
            return torch.mean(first_derivative_last ** 2)
        
        last_transition_point_loss = torch.nn.functional.relu(finite_coordinates[:,-1,0] - self.t_range[1]).mean()
        # print(last_transition_point_loss)


        if finite_coordinates.shape[1] <= 1:
            return 0 + last_transition_point_loss
        elif finite_coordinates.shape[1] <=2:
            return last_derivative_loss() * self.dis_loss_coeff_2 + last_transition_point_loss
        else:
            return derivative_difference_loss() * self.dis_loss_coeff_1 + last_derivative_loss() * self.dis_loss_coeff_2 + last_transition_point_loss

    def loss_last_transition_point(self, finite_coordinates):
        if self.infinite_composition:
            return self.last_loss_coeff * torch.nn.functional.relu(finite_coordinates[:,-1,0] - self.t_range[1]).mean()
        else:
            return self.last_loss_coeff * torch.nn.functional.relu(finite_coordinates[:,-2,0] - (self.t_range[1] - MIN_TRANSITION_POINT_SEP)).mean()
    
    def loss_from_soft_constraint(self, finite_coordinates, derivatives, print_properties):
        """
        finite_coordinates is of shape (batch_size, n_all_coordinates, 2)
        derivatives is a dictionary with keys ('start',1), ('end',1), ('end',2) and values of shape (batch_size,)
        print_properties is of shape (batch_size, n_properties)
        """
        return self.soft_constraint(finite_coordinates, derivatives, print_properties)

    
    def loss(self, X, B, T, Y, with_derivative_loss=True, X0=None, dtw=False):
        """
        Compute the loss function

        Args:
        X: input tensor of shape (batch_size, 1)
        B: pair of input tensors of shape (batch_size, n_cont_features, n_basis_functions) and (batch_size, sum(cat_n_unique_dict))
        T: input tensor of shape (batch_size, n_time_points)
        Y: input tensor of shape (batch_size, n_time_points)
        """

        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(B, X0=X0)
       
        Y_pred = self._forward(all_coefficients, finite_coordinates, B, T, X0=X0) # shape (batch_size, n_time_points)

        if dtw:
            criterion = SoftDTW(gamma=1.0, normalize=True)
            TY = torch.stack([T, Y], dim=2)
            TY_pred = torch.stack([T, Y_pred], dim=2)
            loss = criterion(TY_pred, TY).mean()
        else:
            loss_per_sample = torch.sum((Y_pred - Y) ** 2, dim=1) / Y_pred.shape[1] # shape (batch_size,)
            loss = torch.mean(loss_per_sample)
        total_loss = loss
        if with_derivative_loss:
            derivative_loss = self.loss_discontinuity_of_derivatives(all_coefficients, finite_coordinates)
            total_loss += derivative_loss
            total_loss += self.loss_last_transition_point(finite_coordinates)
            if self.soft_constraint is not None:
                derivatives = {}
                derivatives[('start',1)] = self._boundary_derivative('start',1,B,finite_coordinates,all_coefficients)
                derivatives[('end',1)] = self._boundary_derivative('end',1,B,finite_coordinates,all_coefficients)
                derivatives[('end',2)] = self._boundary_derivative('end',2,B,finite_coordinates,all_coefficients)
                raw_properties = self._infinite_properties_from_weights(B)
                print_properties = self._extract_print_properties_infinite_motif(raw_properties,finite_coordinates,derivatives[('end',1)],derivatives[('end',2)])
                total_loss += self.loss_from_soft_constraint(finite_coordinates, derivatives, print_properties)
       
        return total_loss
    