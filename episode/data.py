import pandas as pd
import numpy as np



class Dataset:

    def __init__(self, name, V, T, Y, t_range, x0=None):
        """
        Initializes the dataset with the given parameters.

        Args:
            name (str): The name of the dataset.
            V (pandas.DataFrame): The features of the dataset (D x K). Categorical features should also have type "category").
            T (list of numpy arrays): The time points at which the trajectories are measured. List has D numpy arrays; each array has shape N_d.
            Y (list of numpy arrays): The observed values of the trajectories. List has D numpy arrays; each array has shape N_d x M.
            x0 (dict, optional): dictionary of initial conditions for each trajectory if available in V. (m:k) m is the index of the trajectory and k is the index of the feature.

            If N_d = N for all d, then you can also pass numpy arrays for T and Y.
            x0 may not be accurate after one-hot encoding of categorical features.
        """
        self.name = name
        self.V = V # features, D x K
        self.T = T # time points, list of D numpy arrays, each array has shape N_d
        self.Y = Y # observed values, list of D numpy arrays, each array has shape N_d x M
        self.t_range = t_range
        self.x0 = x0 # initial conditions, dict of (m:k) m is the index of the trajectory and k is the index of the feature.

        self.categorical_features_indices = self._get_categorical_features_indices()
        self.continuous_features_indices = [i for i in range(V.shape[1]) if i not in self.categorical_features_indices]

    def get_t_grid(self):
        return self.T[0]

    def get_M(self):
        """
        Returns the dimension of the trajectory.
        """
        return self.Y[0].shape[1]
    
    @property
    def M(self):
        return self.get_M()
    
    def _get_categorical_features_indices(self):
        """
        Returns the indices of the categorical features in the dataset.
        """
        return [i for i, col in enumerate(self.V.columns) if self.V[col].dtype.name == 'category']

    def get_V(self, one_hot_encode=False, ordinal_encode=False, return_numpy=False):
        """
        Retrieve the dataset `V`, with an option to one-hot encode categorical features.

        Parameters:
        one_hot_encode (bool): If True, returns the dataset with categorical features one-hot encoded. 
                               If False, returns the dataset as is.

        Returns:
        pd.DataFrame: The dataset `V`, optionally one-hot encoded.
        """
        V = self.V.copy()
        if one_hot_encode and ordinal_encode:
            raise ValueError("Cannot one-hot encode and ordinal encode at the same time.")
        if one_hot_encode:
            V = pd.get_dummies(V, columns=self.V.columns[self.categorical_features_indices], drop_first=True)
        elif ordinal_encode:
            # Encode the categorical features as integers
            for i in self.categorical_features_indices:
                V.iloc[:,i] = V.iloc[:,i].cat.codes

        if return_numpy:
            return V.astype(np.float64).to_numpy()
        return V
    
    def get_T(self):
        """
        Returns the time points at which the trajectories are measured.
        """
        return self.T
    
    def get_Y(self, m=None):
        """
        Args:
            m (int, optional): The dimension of the trajectory for which to return the measurements. If None, return for all dimensions.
        Returns:
            list of numpy arrays: The observed values of the trajectories. List has D numpy arrays; each array has shape N_d x M if m is None, else N_d.
        """
        if m is None:
            return self.Y
        
        return [y[:,m] for y in self.Y]

    def __repr__(self):
        return f"Dataset({self.name})"
    
    def get_V_T_Y(self, one_hot_encode=False, ordinal_encode=False, return_numpy=False, m=None):
        return self.get_V(one_hot_encode, ordinal_encode, return_numpy), self.get_T(), self.get_Y(m)
    
    def _create_subset(self, indices, suffix=""):

        V = self.V.iloc[indices]
        T = [self.T[i] for i in indices]
        Y = [self.Y[i] for i in indices]

        return Dataset(self.name + "__" + suffix, V, T, Y, self.t_range, x0=self.x0)

    
    def split(self, train_size=0.8, val_size=None, seed=0):
        """
        Splits the dataset into training, validation and test sets.

        Args:
            train_size (float): The proportion of the dataset to include in the training set.
            val_size (float, optional): The proportion of the dataset to include in the validation set. 
                                        If None, the validation set is not created, only test set is created.
            seed (int, optional): The seed to use for the random number generator.
        
        Returns:
            tuple: A tuple containing the training, validation and test sets or only the training and test sets if val_size is None.
        """

        # Check if the sizes are valid
        if train_size < 0 or train_size > 1:
            raise ValueError("train_size should be between 0 and 1")
        if val_size is not None and (val_size < 0 or val_size > 1):
            raise ValueError("val_size should be between 0 and 1")
        if val_size is not None and train_size + val_size > 1:
            raise ValueError("train_size + val_size should be less than or equal to 1")

        # Create numpy random generator
        rng = np.random.default_rng(seed)

        # Get the number of trajectories
        D = len(self.V)

        # Get the indices of the trajectories
        indices = np.arange(D)

        # Shuffle the indices
        rng.shuffle(indices)

        # Get the number of trajectories in the training set
        n_train = int(train_size * D)

        # Get the indices of the training set
        train_indices = indices[:n_train]

        if val_size is not None:
            # Get the number of trajectories in the validation set
            n_val = int(val_size * D)

            # Get the indices of the validation set
            val_indices = indices[n_train:n_train+n_val]

            # Get the indices of the test set
            test_indices = indices[n_train+n_val:]

            # Create the training, validation and test sets
            train_set = self._create_subset(train_indices)
            val_set = self._create_subset(val_indices)
            test_set = self._create_subset(test_indices)

            return train_set, val_set, test_set
        else:
            # Get the indices of the test set
            test_indices = indices[n_train:]

            # Create the training and test sets
            train_set = self._create_subset(train_indices)
            test_set = self._create_subset(test_indices)

            return train_set, test_set
    
    def get_name(self):
        return self.name
    
    def __len__(self):
        return self.V.shape[0]
    
    def get_M(self):
        return self.Y[0].shape[1]
    
    def get_K(self,on_hot_encode=False):
        if on_hot_encode:
            return self.get_V(one_hot_encode=True).shape[1]
        return self.V.shape[1]