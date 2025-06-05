from episode import utils
from episode.gam import GAM, CustomPropertyFunction, ShapeFunction, ZeroPropertyFunction, NaNPropertyFunction
import numpy as np
import torch


class SinglePropertyMap:

    def __init__(self, 
                 composition,
                 transition_point_predictor,
                 derivative_predictor,
                 infinite_motif_predictor):
        
        """
        Args:
        composition: a tuple of strings representing the composition
        transition_point_predictor: a dictionary with keys (transition_point_index: int,coordinate: {'x','t'}) and values of type GAM
        derivative_predictor: a dictionary with keys (boundary: {'start','end'},order: {1,2}) and values of type GAM
        infinite_motif_predictor: a list of GAM
        """
        
        self.composition = composition
        self.transition_point_predictor = transition_point_predictor
        self.derivative_predictor = derivative_predictor
        self.infinite_motif_predictor = infinite_motif_predictor
        self.n_transition_points = len(transition_point_predictor) // 2

    def predict_transition_point(self,V,transition_point_index,coordinate,reduce=True):
        """
        Predict the value of the transition point

        Args:
        V: a numpy array of shape (batch_size, n_features)
        transition_point_index: an integer
        coordinate: a string, either 'x' or 't'

        Returns:
        a numpy array of shape (batch_size,)
        """
        
        return self.transition_point_predictor[(transition_point_index,coordinate)].predict(V,reduce=reduce)
    
    def predict_derivative(self,V,boundary,order,reduce=True):
        """
        Predict the value of the derivative

        Args:
        V: a numpy array of shape (batch_size, n_features)
        boundary: a string, either 'start' or 'end'
        order: an integer, either 1 or 2

        Returns:
        a numpy array of shape (batch_size,)
        """
        predictor = self.derivative_predictor[(boundary,order)]
        if isinstance(predictor,GAM):
            if self.composition[0][-1] != 'c': # If the first motif is infinite
                if boundary == 'end':
                    # if order == 1:
                    #     sign = 1 if self.composition[-1][0] == '+' else -1
                    # elif order == 2:
                    #     sign = 1 if self.composition[-1][1] == '+' else -1
                    # return sign * predictor.predict(V,reduce=False)
                    return predictor.predict(V,reduce=reduce)
                elif boundary == 'start':
                    # This should not happen
                    return np.ones(V.shape[0]) * np.nan
            if boundary == 'start':
                motif_index = 0
                side = "left"
            elif boundary == 'end':
                # the end derivative is predicted by a GAM only if there is only one motif (covered earlier)
                # or if the composition is finite (but not with a single motif)
                motif_index = len(self.composition) - 1
                side = "right"
           
            coefficients = utils.get_first_derivative_range_coefficients(self.composition,motif_index,side)
            point1_t = self.transition_point_predictor[(motif_index,'t')].predict(V,reduce=False)
            point1_x = self.transition_point_predictor[(motif_index,'x')].predict(V,reduce=False)
            point2_t = self.transition_point_predictor[(motif_index+1,'t')].predict(V,reduce=False)
            point2_x = self.transition_point_predictor[(motif_index+1,'x')].predict(V,reduce=False)
            slope = (point2_x - point1_x) / (point2_t - point1_t)
            slope_min = slope * coefficients[0]
            slope_max = slope * coefficients[1]
            return slope_min + utils.sigmoid(predictor.predict(V,reduce=False)) * (slope_max - slope_min)
        else:
            return predictor.predict(V,reduce=reduce)
    
    def predict_infinite_motif(self,V,index,reduce=True):
        """
        Predict the value of the infinite motif

        Args:
        V: a numpy array of shape (batch_size, n_features)
        index: an integer representing the index of the property

        Returns:
        a numpy array of shape (batch_size,)
        """
        last_finite_tp = len(self.composition) - 1
        infinite_motif = self.composition[-1]
        if infinite_motif == '++f':
            return 1 / self.infinite_motif_predictor[index].predict(V,reduce=reduce)
        elif infinite_motif == '+-p':
            return self.infinite_motif_predictor[index].predict(V,reduce=reduce)
        elif infinite_motif == '+-h':
            if index == 0:
                return self.infinite_motif_predictor[index].predict(V,reduce=reduce)
            elif index == 1:
                y0 = self.predict_transition_point(V,last_finite_tp,'x',reduce=reduce)
                y1 = self.predict_derivative(V,'end',1,reduce=reduce)
                c =  self.infinite_motif_predictor[0].predict(V,reduce=reduce)
                offset = np.log(3)*(c-y0)/(2*y1)
                return offset + self.infinite_motif_predictor[index].predict(V,reduce=reduce)
        elif infinite_motif == '-+f':
            return self.infinite_motif_predictor[index].predict(V,reduce=reduce)
        elif infinite_motif == '-+h':
            if index == 0:
                return self.infinite_motif_predictor[index].predict(V,reduce=reduce)
            elif index == 1:
                y0 = self.predict_transition_point(V,last_finite_tp,'x',reduce=reduce)
                y1 = self.predict_derivative(V,'end',1,reduce=reduce)
                c =  self.infinite_motif_predictor[0].predict(V,reduce=reduce)
                offset = np.log(3)*(y0-c)/(-2*y1)
                return offset + self.infinite_motif_predictor[index].predict(V,reduce=reduce)
        elif infinite_motif == '--f':
            return 1 / self.infinite_motif_predictor[index].predict(V,reduce=reduce)
        
    
    def predict_all_transition_points(self,V,reduce=True):
        """
        Predict the values of all transition points

        Args:
        V: a numpy array of shape (batch_size, n_features)

        Returns:
        a numpy array of shape (batch_size, n_transition_points, 2)
        """
        if len(V.shape) == 1:
            V = V.reshape(1,-1)
        else:
            reduce = False
        results = np.zeros((V.shape[0],self.n_transition_points,2))
        for i in range(self.n_transition_points):
            results[:,i,0] = self.predict_transition_point(V,i,'t',reduce=False)
            results[:,i,1] = self.predict_transition_point(V,i,'x',reduce=False)
        
        if reduce:
            results = results[0]
        
        return results
    
    def predict_all_infinite_motif_properties(self,V,reduce=True):
        """
        Predict the values of all infinite motif properties

        Args:
        V: a numpy array of shape (batch_size, n_features)

        Returns:
        a numpy array of shape (batch_size, n_infinite_motif_properties)
        """
        if len(V.shape) == 1:
            V = V.reshape(1,-1)
        else:
            reduce = False

        if self.infinite_motif_predictor is None:
            return np.full((V.shape[0],1),np.nan)
        
        results = np.zeros((V.shape[0],len(self.infinite_motif_predictor)))
        for i in range(len(self.infinite_motif_predictor)):
            results[:,i] = self.predict_infinite_motif(V,i,reduce=False)
        
        if reduce:
            results = results.squeeze()
        
        return results

class PropertyMapExtractor:

    def __init__(self,config,basis_functions,one_hot_basis_functions):
        self.config = config
        self.basis_functions = basis_functions
        self.one_hot_basis_functions = one_hot_basis_functions
        self.x0_included = self.config['x0_included']

    def _get_tensors(self,*args):
        """
        Convert numpy arrays to torch tensors

        Args:
        args: numpy arrays

        Returns:
        tuple of torch tensors
        """
        torch_device = utils.get_torch_device(self.config['device'])
        return tuple([torch.tensor(arg, dtype=torch.float32, device=torch_device) for arg in args])
    

    
    def _get_b_X0_tensors(self,x,feature_index,specific_V_ranges):
        X0_tensor = None
        if self.x0_included:
            if isinstance(self.config['x0_index'],int):
                if feature_index == self.config['x0_index']:
                    X0_tensor = self._get_tensors(x.reshape(-1,1))[0]
            elif isinstance(self.config['x0_index'],float):
                X0_tensor = self._get_tensors(np.full((len(x),1),self.config['x0_index']))[0]
            else:
                raise ValueError("Invalid x0_index")
            
        X = np.zeros((len(x),len(specific_V_ranges)))
        X[:,feature_index] = x
        if feature_index in self.config['categorical_features_indices']:
            b = self.one_hot_basis_functions.compute_single(X,specific_V_ranges,feature_index)
        else:
            B = self.basis_functions.compute(X,specific_V_ranges)
            cont_feature_index = utils.from_feature_index_to_cont_feature_index(feature_index,self.config['categorical_features_indices'])
            b = B[:,cont_feature_index,:]
        b_tensor = self._get_tensors(b)[0]

        return b_tensor,X0_tensor
    
    def _get_B_X0_tensors(self,X,specific_V_ranges):
        B = self.basis_functions.compute(X,specific_V_ranges)
        if len(self.config['categorical_features_indices']) > 0:
            B_cat, _ = self.one_hot_basis_functions.compute(X,specific_V_ranges)
        else:
            B_cat = np.zeros_like(X)[:,[0]]

        B_tensor, B_cat_tensor = self._get_tensors(B, B_cat)
        if self.x0_included:
            if isinstance(self.config['x0_index'],int):
                X0_tensor = self._get_tensors(X[:,[self.config['x0_index']]])[0]
            elif isinstance(self.config['x0_index'],float):
                X0_tensor = self._get_tensors(np.full((len(X),1),self.config['x0_index']))[0]
            else:
                raise ValueError("Invalid x0_index")
        else:
            X0_tensor = None
        B = (B_tensor,B_cat_tensor)

        return B,X0_tensor

    def _extract_shape_function_of_transition_point(self, torch_model, ind, coordinate, feature_index, specific_V_ranges, histogram=None, categories=None, name=None):

        def shape_function(x):
            b_tensor,X0_tensor = self._get_b_X0_tensors(x,feature_index,specific_V_ranges)
            torch_model.eval()
            with torch.no_grad():
                pred = torch_model.extract_shape_function_of_transition_point(ind,coordinate,feature_index,b_tensor,X0_tensor).detach().cpu()
            return pred.numpy()
        
        return ShapeFunction(shape_function,specific_V_ranges[feature_index], categories=categories, histogram=histogram, name=name)
    
    def _extract_bias_of_transition_point(self, torch_model, ind, coordinate):

        torch_model.eval()
        with torch.no_grad():
            pred = torch_model.extract_bias_of_transition_point(ind,coordinate).detach().cpu()
        return pred.numpy()
    
    def _extract_gam_of_transition_point(self, torch_model, ind, coordinate, specific_V_ranges, histograms=None, all_categories={}, feature_names=None):

        n_features = len(specific_V_ranges)
        
        shape_functions = []
        for i in range(n_features):
            if i in all_categories:
                categories = all_categories[i]
            else:
                categories = None
            if histograms is not None:
               histogram = histograms[i]
            else:
                histogram = None
            if feature_names is not None:
                name = feature_names[i]
            else:
                name = None
            shape_functions.append(self._extract_shape_function_of_transition_point(torch_model, ind, coordinate, i, specific_V_ranges, histogram=histogram, categories=categories, name=name))
        bias = self._extract_bias_of_transition_point(torch_model, ind, coordinate)
        return GAM(shape_functions,bias)
    
    def _extract_shape_function_of_infinite_property(self, torch_model, ind, feature_index, specific_V_ranges, categories=None,histogram=None, name=None):

        def shape_function(x):
            b_tensor,X0_tensor = self._get_b_X0_tensors(x,feature_index,specific_V_ranges)
            torch_model.eval()
            with torch.no_grad():
                pred = torch_model.extract_shape_function_of_infinite_property(ind,feature_index,b_tensor,X0_tensor).detach().cpu()
            return pred.numpy()
        
        return ShapeFunction(shape_function,specific_V_ranges[feature_index], categories=categories, histogram=histogram, name=name)
    
    def _extract_bias_of_infinite_property(self, torch_model, ind):

        torch_model.eval()
        with torch.no_grad():
            pred = torch_model.extract_bias_of_infinite_property(ind).detach().cpu()
        return pred.numpy()
    
    def _extract_gam_of_infinite_property(self, torch_model, ind, specific_V_ranges, histograms=None, all_categories={}, feature_names=None):

        n_features = len(specific_V_ranges)

        shape_functions = []
        for i in range(n_features):
            if i in all_categories:
                categories = all_categories[i]
            else:
                categories = None
            if histograms is not None:
               histogram = histograms[i]
            else:
                histogram = None
            if feature_names is not None:
                name = feature_names[i]
            else:
                name = None
            shape_functions.append(self._extract_shape_function_of_infinite_property(torch_model, ind, i, specific_V_ranges, categories=categories, histogram=histogram, name=name))
        bias = self._extract_bias_of_infinite_property(torch_model, ind)
        return GAM(shape_functions,bias)
    
    def _extract_shape_function_of_derivative(self, torch_model, boundary, order, feature_index, specific_V_ranges, categories=None,histogram=None, name=None): 

        def shape_function(x):
            b_tensor,X0_tensor = self._get_b_X0_tensors(x,feature_index,specific_V_ranges)
            torch_model.eval()
            with torch.no_grad():
                pred = torch_model.extract_shape_function_of_derivative(boundary,order,feature_index,b_tensor,X0_tensor).detach().cpu()
            return pred.numpy()
        
        return ShapeFunction(shape_function,specific_V_ranges[feature_index], categories=categories, histogram=histogram, name=name)
    
    def _extract_bias_of_derivative(self, torch_model, boundary, order):
        
        torch_model.eval()
        with torch.no_grad():
            pred = torch_model.extract_bias_of_derivative(boundary,order).detach().cpu()
        return pred.numpy()

    def _extract_gam_of_derivative(self, torch_model, boundary, order,specific_V_ranges, histograms=None, all_categories={}, feature_names=None):

        n_features = len(specific_V_ranges)

        shape_functions = []

        for i in range(n_features):
            if i in all_categories:
                categories = all_categories[i]
            else:
                categories = None
            if histograms is not None:
               histogram = histograms[i]
            else:
                histogram = None
            if feature_names is not None:
                name = feature_names[i]
            else:
                name = None

            shape_functions.append(self._extract_shape_function_of_derivative(torch_model, boundary, order, i,specific_V_ranges, categories=categories, histogram=histogram, name=name))
        bias = self._extract_bias_of_derivative(torch_model, boundary, order)
        return GAM(shape_functions,bias)
    
    def _construct_derivative_predictor(self,composition,torch_model, specific_V_ranges, histograms=None, all_categories={}, feature_names=None):

        derivative_predictor = {}
        
        first_derivative_at_start_status = utils.get_first_derivative_at_start_status(composition)
        first_derivative_at_end_status = utils.get_first_derivative_at_end_status(composition)
        second_derivative_at_end_status = utils.get_second_derivative_at_end_status(composition)

        # boundary = 'start' and order = 1
        if first_derivative_at_start_status == 'weights':
            derivative_predictor[('start',1)] = self._extract_gam_of_derivative(torch_model, 'start', 1, specific_V_ranges, histograms, all_categories, feature_names)
        elif first_derivative_at_start_status == 'none':
            derivative_predictor[('start',1)] = NaNPropertyFunction()

        # boundary = 'end' and order = 1
        if first_derivative_at_end_status == 'weights':
            derivative_predictor[('end',1)] = self._extract_gam_of_derivative(torch_model, 'end', 1, specific_V_ranges, histograms, all_categories, feature_names)
        elif first_derivative_at_end_status == 'zero':
            derivative_predictor[('end',1)] = ZeroPropertyFunction()
        elif first_derivative_at_end_status == 'cubic':

            def property_function(X):
                B,X0_tensor = self._get_B_X0_tensors(X,specific_V_ranges)
                torch_model.eval()
                with torch.no_grad():
                    pred = torch_model.extract_boundary_derivative('end',1,B,X0=X0_tensor).detach().cpu().numpy()
                return pred
            
            derivative_predictor[('end',1)] = CustomPropertyFunction(property_function)
        
        # boundary = 'end' and order = 2
        if second_derivative_at_end_status == 'weights':
            derivative_predictor[('end',2)] = self._extract_gam_of_derivative(torch_model, 'end', 2, specific_V_ranges, histograms, all_categories, feature_names)
        if second_derivative_at_end_status == 'zero':
            derivative_predictor[('end',2)] = ZeroPropertyFunction()
        elif second_derivative_at_end_status == 'cubic':

            def property_function(X):
                B,X0_tensor = self._get_B_X0_tensors(X,specific_V_ranges)
                torch_model.eval()
                with torch.no_grad():
                    pred = torch_model.extract_boundary_derivative('end',2,B,X0=X0_tensor).detach().cpu().numpy()
                return pred
            derivative_predictor[('end',2)] = CustomPropertyFunction(property_function)
        
        return derivative_predictor

    def construct_single_property_map(self,composition,torch_model,specific_V_ranges, histograms=None):

        all_categories = self.config['categorical_features_categories']
        feature_names = self.config['feature_names']

        n_motifs = len(composition)
        infinite_composition = (composition[-1][2] != 'c')
        if infinite_composition:
            n_transition_points = n_motifs
        else:
            n_transition_points = n_motifs + 1

        transition_point_predictor = {}
        for i in range(n_transition_points):
            for coordinate in ['x','t']:
                gam = self._extract_gam_of_transition_point(torch_model, i, coordinate,specific_V_ranges,histograms=histograms, all_categories=all_categories, feature_names=feature_names)
                transition_point_predictor[(i,coordinate)] = gam
        
        derivative_predictor = self._construct_derivative_predictor(composition,torch_model,specific_V_ranges,histograms=histograms, all_categories=all_categories, feature_names=feature_names)

        if infinite_composition:
            infinite_motif_predictor = []
            for i in range(torch_model.number_of_properties_for_infinite_motif()):
                gam = self._extract_gam_of_infinite_property(torch_model, i, specific_V_ranges,histograms=histograms, all_categories=all_categories, feature_names=feature_names)
                infinite_motif_predictor.append(gam)
        else:
            infinite_motif_predictor = None
        
        return SinglePropertyMap(composition,transition_point_predictor,derivative_predictor,infinite_motif_predictor)
