import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
class PropertyFunction(ABC):

    @abstractmethod
    def predict(self,X,reduce=True):
        pass

class ShapeFunction:

    def __init__(self,function,x_range, categories=None, histogram=None, name=None):
        self.function = function
        self.x_range = x_range
        self.categories = categories
        self.histogram = histogram
        self.name = name

        if isinstance(self.x_range,tuple):
            self.categorical = False
        else:
            self.categorical = True

    def __call__(self,x):
        """
        Args:
        x: a numpy array of shape (batch_size,)
        """
        if np.isscalar(x):
            x = np.array([x])
        if not self.categorical:
            # This is a continuous feature
            x = np.clip(x,self.x_range[0],self.x_range[1])

        return self.function(x)

    def predict(self,x,reduce=True):
        reduce = (reduce and np.isscalar(x))
        result = self.__call__(x)   
        if result.shape[0] == 1 and reduce:
            result = result[0]
        return result
    
    def get_expected_value(self):
        """
        Calculate the expected value of the shape function

        Returns:
        a scalar
        """
        if self.histogram is None:
            raise ValueError('Histogram is not provided')
            # TODO: Implement a method to calculate the expected value without the histogram (assuming uniform distribution)
        
        expected_value = 0
        all_points = 0
        if self.categorical:
            x = np.array(sorted(list(self.x_range)))
            pred = self.predict(x)
            for i, value in enumerate(x):
                expected_value += self.histogram[int(value)] * pred[i]
                all_points += self.histogram[int(value)]
            expected_value /= all_points
        else:
            counts, bins = self.histogram
            bin_centers = (bins[:-1] + bins[1:])/2
            pred = self.predict(bin_centers)
            for i, value in enumerate(bin_centers):
                expected_value += counts[i] * pred[i]
                all_points += counts[i]
            expected_value /= all_points
       
        return expected_value
    
    def maximum_value(self):
        """
        Calculate the maximum value of the shape function

        Returns:
        a scalar
        """
        if self.categorical:
            x = np.array(sorted(list(self.x_range)))
        else:
            x = np.linspace(self.x_range[0],self.x_range[1],100)
        
        pred = self.predict(x)
        return np.max(pred)
    
    def minimum_value(self):
        """
        Calculate the minimum value of the shape function

        Returns:
        a scalar
        """
        if self.categorical:
            x = np.array(sorted(list(self.x_range)))
        else:
            x = np.linspace(self.x_range[0],self.x_range[1],100)
        
        pred = self.predict(x)
        return np.min(pred)
    
    def range(self):
        """
        Calculate the range of the shape function

        Returns:
        a tuple (min, max)
        """
        return self.minimum_value(), self.maximum_value()

    def is_constant(self):
        """
        Check if the shape function is constant

        Returns:
        a boolean
        """
        min_value, max_value = self.range()
        return min_value == max_value
    
    def visualize(self, fig=None):
        """
        Creates a visualization of the shape function. It consists of two axes objects: one for the shape function and one for the histogram of the data.
        
        Returns:
        a figure object
        """
        name = self.name

        if fig is None:
            fig = plt.figure(figsize=(3,3))
        
        axs = fig.subplots(2,1,height_ratios=[5,1], sharex=True)

        if self.categorical:
            x = np.array(sorted(list(self.x_range)))
        else:
            x = np.linspace(self.x_range[0],self.x_range[1],100)

        Ey = self.get_expected_value()
        y = self.predict(x) - Ey
        
        if self.categorical:
            axs[0].bar(x,y)
        else:
            axs[0].plot(x,y)

        axs[0].set_ylabel('Shape function')

        if self.categorical:
            # Map xticks to categories
            axs[0].set_xticks(x)
            if self.categories is not None:
                category_labels = [self.categories[int(i)] for i in x] # Not all categories may be present for that property map
            else:
                category_labels = x

            axs[0].set_xticklabels(category_labels)

        if self.histogram is not None:
            if self.categorical:
                values = [self.histogram[int(i)] for i in x]
                axs[1].bar(x,values)   
            else:
                axs[1].stairs(self.histogram[0], self.histogram[1], fill=True)

            axs[1].set_ylabel('Frequency')
        
        if name is not None:
            axs[1].set_xlabel(name)
        else:
            axs[1].set_xlabel("Feature value")

        # fig.tight_layout()
        fig.subplots_adjust(hspace=0.3)

        return fig


class GAM(PropertyFunction):

    def __init__(self,shape_functions,bias):
        """
        
        Args:
        shape_functions: a list of ShapeFunction, each shape function is a function that takes a numpy array
        bias: a scalar
        """
        self.shape_functions = shape_functions
        self.bias = bias
        self.n_shape_functions = len(shape_functions)
 
    def _validate_input(self,X):
        if X.shape[1] != len(self.n_shape_functions):
            raise ValueError('Number of features must be equal to the number of shape functions')
    
    def predict(self,X,reduce=True):
        """
        Predict the value of the GAM

        Args:
        X: a numpy array of shape (batch_size, n_features)

        Returns:
        a numpy array of shape (batch_size,)
        """
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        else:
            reduce = False

        values = np.zeros((X.shape[0],self.n_shape_functions))
        for i in range(self.n_shape_functions):
            x = X[:,i]
            values[:,i] = self.shape_functions[i].predict(x)
        
        result = np.sum(values,axis=1) + self.bias
        
        if result.shape[0] == 1 and reduce:
            result = result[0]
        
        return result
    
    def prune(self,threshold=1e-5):
        """
        Prune the shape functions whose range is smaller than the threshold
        """
        for shape_function in self.shape_functions:
            min_value, max_value = shape_function.range()
            if max_value - min_value < threshold:
                self.bias += shape_function.get_expected_value()
                shape_function.function = lambda x: np.zeros_like(x)
            
    
    def _calculate_size_of_the_grid(self, n_shape_functions_to_plot, max_n_columns=4):
        """
        Calculate the size of the grid for the visualization of the shape functions

        Args:
        n_features: the number of features

        Returns:
        a tuple (n_rows, n_columns)
        """
        all_plots = n_shape_functions_to_plot + 1
        n_rows = all_plots // max_n_columns
        if all_plots % max_n_columns != 0:
            n_rows += 1
        n_columns = min(all_plots,max_n_columns)
        return n_rows, n_columns
    
    def visualize(self, max_n_columns=4, skip_constant=False):

        if skip_constant:
            should_skip = [shape_function.is_constant() for shape_function in self.shape_functions]
        else:
            should_skip = [False] * self.n_shape_functions

        n_shape_functions_to_plot = sum([not skip for skip in should_skip])

        n_rows, n_columns = self._calculate_size_of_the_grid(n_shape_functions_to_plot, max_n_columns)

        # Create the figure from subfigures (using fig.subfigures)
        fig = plt.figure(figsize=(3*n_columns,3*n_rows))
        subfigs = np.atleast_2d(fig.subfigures(n_rows,n_columns, wspace=0.3))
        bias = self.bias * 1
        counter = 0
        for i in range(self.n_shape_functions):
            shape_function = self.shape_functions[i]
            bias += shape_function.get_expected_value()
            if not should_skip[i]:
                subfig = subfigs[counter//n_columns,counter%n_columns]
                shape_function.visualize(subfig)
                counter += 1
        
        subfig = subfigs[counter//n_columns,counter%n_columns]
        axs = subfig.subplots(2,1, height_ratios=[5,1])
        axs[0].plot(np.linspace(0,1,100),bias*np.ones(100))
        axs[0].set_ylabel('Bias')
        # Remove the xticks and xticklabels
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])
        axs[1].set_xticks([])
        axs[1].set_xticklabels([])
        axs[1].set_yticks([])
        axs[1].set_yticklabels([])
    
class ZeroPropertyFunction(PropertyFunction):

    def predict(self,X,reduce=True):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        else:
            reduce = False
        n_samples = X.shape[0]
        result = np.zeros((n_samples,))
        if result.shape[0] == 1 and reduce:
            result = result[0]
        return result
    
class CustomPropertyFunction(PropertyFunction):

    def __init__(self,function):
        self.function = function
    
    def predict(self, X, reduce=True):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        else:
            reduce = False

        result = self.function(X)
        if result.shape[0] == 1 and reduce:
            result = result[0]
        return result
    
class NaNPropertyFunction(PropertyFunction):

    def predict(self,X,reduce=True):
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
        else:
            reduce = False
        n_samples = X.shape[0]
        result = np.full((n_samples,),np.nan)
        if result.shape[0] == 1 and reduce:
            result = result[0]
        return result



