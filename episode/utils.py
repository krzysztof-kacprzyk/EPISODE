import numpy as np
from episode.infinite_motifs import get_default_motif_class
import torch

def sigmoid(x, threshold=20.0):
    maskg20 = x > threshold
    masklm20 = x < -threshold
    res = np.zeros_like(x)
    res[maskg20] = 1.0
    res[masklm20] = 0.0
    res[~maskg20 & ~masklm20] = 1 / (1 + np.exp(-x[~maskg20 & ~masklm20]))
    return res

def softmax(x, temperature=1.0):
    """
    Compute the softmax of vector x with a temperature parameter.
    
    Parameters:
    x (numpy.ndarray): Input data.
    temperature (float): Temperature parameter for scaling.
    
    Returns:
    numpy.ndarray: Softmax probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    
    x = np.array(x) / temperature
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    return e_x / sum_e_x

def softmax_per_row(X, temperature=1.0):
    if len(X.shape) == 1:
        X = X.reshape(1,-1)
    for row_id in range(X.shape[0]):
        X[row_id,:] = softmax(X[row_id,:])

    return X


def type_of_transition_point(composition, ind):
    composition = composition
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



def get_first_derivative_range_coefficients(composition, motif_index, which_point):
    motif = composition[motif_index]
                                
    if which_point == 'left':
        if motif == '++c':
            if type_of_transition_point(composition, motif_index+1) == 'end':
                return (0,1)
            elif type_of_transition_point(composition, motif_index+1) == 'inflection':
                return (0,1)
        elif motif == "+-c":
            if type_of_transition_point(composition, motif_index+1) == 'end':
                return (1,2)
            elif type_of_transition_point(composition, motif_index+1) == 'max':
                return (1.5,3)
            elif type_of_transition_point(composition, motif_index+1) == 'inflection':
                return (1,3)
        elif motif == "-+c":
            if type_of_transition_point(composition, motif_index+1) == 'end':
                return (1,2)
            elif type_of_transition_point(composition, motif_index+1) == 'min':
                return (1.5,3)
            elif type_of_transition_point(composition, motif_index+1) == 'inflection':
                return (1,3)
        elif motif == "--c":
            if type_of_transition_point(composition, motif_index+1) == 'end':
                return (0,1)
            elif type_of_transition_point(composition, motif_index+1) == 'inflection':
                return (0,1)
    
    elif which_point == 'right':
        if motif == '++c':
            if type_of_transition_point(composition, motif_index) == 'inflection':
                return (1,3)
            elif type_of_transition_point(composition, motif_index) == 'min':
                return (1.5,3)
        elif motif == "+-c":
            if type_of_transition_point(composition, motif_index) == 'inflection':
                return (0,1)
        elif motif == "-+c":
            if type_of_transition_point(composition, motif_index) == 'inflection':
                return (0,1)
        elif motif == "--c":
            if type_of_transition_point(composition, motif_index) == 'inflection':
                return (1,3)
            elif type_of_transition_point(composition, motif_index) == 'max':
                return (1.5,3)  

def is_bounded_motif(motif):
    if motif[2] == 'c':
        return True
    else:
        return False
    
def is_unbounded_motif(motif):
    return not is_bounded_motif(motif)
    

def is_unbounded_composition(composition):
    n_motifs = len(composition)
    return is_unbounded_motif(composition[n_motifs-1])


def get_first_derivative_at_end_status(composition):

    n_motifs = len(composition)

    if is_unbounded_composition(composition):
        if n_motifs == 1:
            # If there is only one infinite motif, then the first derivative at end is specified by the parameter/weights
            return "weights"
        else:
            # If there is a bounded motif before the infinite motif
            index_last_finite_tp = n_motifs - 1
            type_of_last_finite_tp = type_of_transition_point(composition,index_last_finite_tp)

            if type_of_last_finite_tp == 'max' or type_of_last_finite_tp == 'min':
                # It has to be zero
                return "zero"
            elif type_of_last_finite_tp == 'inflection':
                # It is determined by the cubic because the second derivative is fixed at 0
                return "cubic"
    else:
        # If the composition is finite, then the first derivative at end is specified by the parameter/weights
        # unless the composition has only one motif
        if n_motifs == 1:
            return "cubic"
        else:
            return "weights"
    
def get_second_derivative_at_end_status(composition):
    
    n_motifs = len(composition)

    if is_unbounded_composition(composition):
        if n_motifs == 1:
            # If there is only one infinite motif, then the second derivative at end is specified by the parameter/weights
            # unless the unbounded motif does not allow for anyhting but zero
            if get_default_motif_class(composition[0]).second_derivative_vanishes():
                return "zero"
            else:
                return "weights"
        else:
            # If there is a bounded motif before the infinite motif
            index_last_finite_tp = n_motifs - 1
            type_of_last_finite_tp = type_of_transition_point(composition,index_last_finite_tp)

            if type_of_last_finite_tp == 'max' or type_of_last_finite_tp == 'min':
                # Determined by the cubic because the first derivative is fixed at 0
                return "cubic"
            elif type_of_last_finite_tp == 'inflection':
                # It has to be zero
                return "zero"
    else:
        # If the composition if finite then the second derivative at end is specified by the cubic
        # As it is the first derivative at end that determins the cubic
        return "cubic"
    
def get_first_derivative_at_start_status(composition):

    n_motifs = len(composition)

    if is_unbounded_composition(composition):
        if n_motifs == 1:
            # If there is only one infinite motif, then the first derivative at start is none because it coincides with the end
            return "none"
        else:
            # If there is a bounded motif before the infinite motif then the first derivative at start is specified by the parameter/weights
            return "weights"
    else:
        # If the composition is finite, then the first derivative at start is specified by the parameter/weights
        return "weights"

def evaluate_cubic(coefficients, x, derivative_order = 0):
    """
    Evaluate a cubic polynomial at x (or its derivatives).
    
    Parameters:
    coefficients (numpy.ndarray): Coefficients of the cubic polynomial. Should be of shape (4,) or (batch_size,4)
    The coefficients are ordered as [a,b,c,d] where the polynomial is a*x^3 + b*x^2 + c*x + d
    x (numpy.ndarray): Input data. Should be scalar or (batch_size,)
    derivative_order (int): Order of derivative to evaluate.

    Returns:
    numpy.ndarray: Value of the cubic polynomial at x. Should be a scalar or (batch_size,)
    """

    if len(coefficients.shape) == 1:
        coefficients = coefficients.reshape(1,-1)
        is_scalar = True
    else:
        is_scalar = False
    assert coefficients.shape[1] == 4
    if derivative_order == 0:
        result = coefficients[:,0] * x**3 + coefficients[:,1] * x**2 + coefficients[:,2] * x + coefficients[:,3]
    elif derivative_order == 1:
        result = 3 * coefficients[:,0] * x**2 + 2 * coefficients[:,1] * x + coefficients[:,2]
    elif derivative_order == 2:
        result = 6 * coefficients[:,0] * x + 2 * coefficients[:,1]
    elif derivative_order == 3:
        return 6 * coefficients[:,0]
    else:
        raise ValueError("Derivative order must be 0, 1, 2 or 3")
    
    if result.shape[0] == 1 and is_scalar:
        return result[0]
    
def get_torch_device(device):
    if device == 'cuda' or device == 'gpu':
        return torch.device('cuda:0')
    elif device == 'cpu':
        return torch.device('cpu')
    else:
        raise ValueError('Unknown device')

def get_lightning_accelerator(device):
    if device == 'cuda' or device == 'gpu':
        return 'gpu'
    elif device == 'cpu':
        return 'cpu'
    else:
        raise ValueError('Unknown device')
    
def from_feature_index_to_cont_feature_index(feature_index, categorical_features_indices):
        """
        Convert a feature index to a continuous feature index

        Args:
        feature_index: integer, index of the feature

        Returns:
        output integer
        """
        if feature_index in categorical_features_indices:
            return None
        cont_feature_index = 0
        for i in range(feature_index):
            if i not in categorical_features_indices:
                cont_feature_index += 1
        
        return cont_feature_index