from copy import copy
import numpy as np

class TreeNode:
    """
    A simple class representing a node in the decision tree.
    """
    def __init__(self, 
                 feature_index=None, 
                 split_value=None, 
                 left=None, 
                 right=None, 
                 value=None,
                 is_categorical=False):
        """
        Parameters
        ----------
        feature_index : int, optional
            The index of the feature used for splitting.
        split_value : float or any (threshold or category), optional
            If is_categorical=False, this is the numeric threshold.
            If is_categorical=True, this is the specific category.
        left : TreeNode, optional
            Left child node.
        right : TreeNode, optional
            Right child node.
        value : np.ndarray, optional
            The class distribution (probabilities) of the leaf node.
        is_categorical : bool
            Whether the feature at feature_index is categorical.
        """
        self.feature_index = feature_index
        self.split_value = split_value
        self.left = left
        self.right = right
        self.value = value
        self.is_categorical = is_categorical

        self.X_leaf = None
        self.Y_leaf = None


class DecisionTreeClassifier:
    """
    CART Decision Tree for classification with support for:
      - Numeric features
      - Categorical features
      - One-hot encoded labels
    """
    def __init__(self, 
                 max_depth=5, 
                 min_samples_split=2,
                 categorical_features=None,
                 min_gini_impurity=1e-7, metric_name='gini', offset=1e-3, min_samples_leaf=1):
        """
        Parameters
        ----------
        max_depth : int
            The maximum depth of the tree.
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        categorical_features : list of int, optional
            List of feature indices that are categorical. By default, None
            means all features are treated as numeric.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorical_features = set(categorical_features) if categorical_features else set()
        self.root = None
        self.min_gini_impurity = min_gini_impurity
        self.metric_name = metric_name
        if metric_name == 'gini':
            self.metric = self._gini_impurity
        elif metric_name == 'error':
            self.metric = self._min_error
        self.offset = offset
        self.min_samples_leaf = min_samples_leaf

    def _gini_impurity(self, Y):
        """
        Compute the Gini impurity for a set of one-hot encoded labels.
        
        Parameters
        ----------
        Y : np.ndarray of shape (n_samples, n_classes)
        
        Returns
        -------
        float
            The Gini impurity.
        """
        total_samples = Y.shape[0]
        if total_samples == 0:
            return 0.0
        
        class_counts = Y.sum(axis=0)  # sum across samples -> shape: (n_classes,)
        probs = class_counts / total_samples
        return 1.0 - np.sum(probs**2)
    
    def _min_error(self, Y):

        total_samples = Y.shape[0]
        if total_samples == 0:
            return 0.0
        
        error_per_class = Y.mean(axis=0)  # mean across samples -> shape: (n_classes,)
        min_error = np.min(error_per_class)
        return min_error

    def _split_dataset_numeric(self, X, Y, feature_index, threshold):
        """
        Split dataset based on a numeric threshold: <= threshold goes left, else right.
        """
        left_mask = (X[:, feature_index] <= threshold)
        right_mask = ~left_mask
        
        return X[left_mask], Y[left_mask], X[right_mask], Y[right_mask]

    def _split_dataset_categorical(self, X, Y, feature_index, category):
        """
        Split dataset based on a categorical feature: == category goes left, else right.
        """
        left_mask = (X[:, feature_index] == category)
        right_mask = ~left_mask
        
        return X[left_mask], Y[left_mask], X[right_mask], Y[right_mask]

    def _find_best_split(self, X, Y):
        """
        Find the best split (feature_index, split_value) that achieves the lowest Gini impurity.
        For numeric features, split_value is a threshold; for categorical, it is a category.
        
        Returns
        -------
        best_feature : int or None
        best_split_value : float or any, depending on feature type
        is_categorical : bool
            Whether the best feature is categorical.
        """

        n_samples, n_features = X.shape
        if n_samples == 0:
            return None, None, False

        current_impurity = self.metric(Y)
        best_feature = None
        best_split_value = None
        best_impurity = float('inf')
        best_is_categorical = False
        
        for feature_index in range(n_features):
            unique_values = np.unique(X[:, feature_index])
            
            # If the feature is categorical, each unique value is a possible "split_value".
            # If numeric, each unique value is a threshold.
            if feature_index in self.categorical_features:
                # Categorical splitting
                for category in unique_values:
                    X_left, Y_left, X_right, Y_right = self._split_dataset_categorical(
                        X, Y, feature_index, category
                    )
                    if len(Y_left) < self.min_samples_leaf or len(Y_right) < self.min_samples_leaf:
                        continue
                    w_left = len(Y_left) / n_samples
                    w_right = len(Y_right) / n_samples
                    impurity = (w_left * self.metric(Y_left) 
                                + w_right * self.metric(Y_right))
                    
                    if impurity < best_impurity:
                        best_impurity = impurity
                        best_feature = feature_index
                        best_split_value = category
                        best_is_categorical = True
            else:
                # Numeric splitting
                for threshold in unique_values:
                    X_left, Y_left, X_right, Y_right = self._split_dataset_numeric(
                        X, Y, feature_index, threshold
                    )
                    if len(Y_left) < self.min_samples_leaf or len(Y_right) < self.min_samples_leaf:
                        continue
                    w_left = len(Y_left) / n_samples
                    w_right = len(Y_right) / n_samples
                    impurity = (w_left * self.metric(Y_left) 
                                + w_right * self.metric(Y_right))
                    
                    if impurity < best_impurity:
                        best_impurity = impurity
                        best_feature = feature_index
                        best_split_value = threshold
                        best_is_categorical = False
        
        # If no improvement, return None
        if current_impurity - best_impurity <= self.offset * current_impurity:
            return None, None, False

        return best_feature, best_split_value, best_is_categorical

    def _build_tree(self, X, Y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        Y : np.ndarray of shape (n_samples, n_classes)
        depth : int
            Current depth of the tree.
        
        Returns
        -------
        TreeNode
            The root of the subtree.
        """
        n_samples = X.shape[0]
        
        # Stopping conditions
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            self.metric(Y) < self.min_gini_impurity):
            # Leaf node: store the class probabilities
            leaf_value = Y.sum(axis=0) / n_samples  # shape: (n_classes,)
            return TreeNode(value=leaf_value)

        # Find best split
        feature_index, split_value, is_categorical = self._find_best_split(X, Y)
        if feature_index is None:
            # No further split improves impurity
            leaf_value = Y.sum(axis=0) / n_samples
            return TreeNode(value=leaf_value)

        # Split the dataset
        if is_categorical:
            X_left, Y_left, X_right, Y_right = self._split_dataset_categorical(
                X, Y, feature_index, split_value
            )
        else:
            X_left, Y_left, X_right, Y_right = self._split_dataset_numeric(
                X, Y, feature_index, split_value
            )
        
        # Recursively build subtrees
        left_child = self._build_tree(X_left, Y_left, depth + 1)
        right_child = self._build_tree(X_right, Y_right, depth + 1)
        
        # Create the internal node
        return TreeNode(feature_index=feature_index,
                        split_value=split_value,
                        left=left_child,
                        right=right_child,
                        is_categorical=is_categorical)

    def fit(self, X, Y):
        """
        Fit the decision tree to the data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        Y : np.ndarray of shape (n_samples, n_classes)
        """
        self.root = self._build_tree(X, Y)

    def _predict_sample(self, x, node):
        """
        Predict the class probabilities for a single sample using the tree.
        
        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
        node : TreeNode
            The current node in the tree.
        
        Returns
        -------
        np.ndarray of shape (n_classes,)
            The predicted probabilities.
        """
        # Leaf node
        if node.value is not None:
            return node.value
        
        # Internal node: check categorical vs numeric
        if node.is_categorical:
            # Go left if the sample has the same category as split_value
            if x[node.feature_index] == node.split_value:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
        else:
            # Numeric
            if x[node.feature_index] <= node.split_value:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        return np.array([self._predict_sample(row, self.root) for row in X])

    def predict(self, X):
        """
        Predict class labels for each sample in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted class labels (as integers).
        """
        if self.metric_name == 'gini':
            probabilities = self.predict_proba(X)
            return np.argmax(probabilities, axis=1)
        elif self.metric_name == 'error':
            errors = self.predict_proba(X)
            return np.argmin(errors, axis=1)
    
    def print_tree(self, node=None, depth=0):
        """
        Recursively print the structure of the decision tree.

        Parameters
        ----------
        node : TreeNode, optional
            The current node in the tree. If None, starts from the root.
        depth : int
            Current depth of the tree, used for indentation.
        """
        # If no node is provided, start from the root of the tree
        if node is None:
            node = self.root

        indent = "  " * depth  # Indentation based on depth

        # If it's a leaf node, display class distribution
        if node.value is not None:
            if self.metric == 'gini':
                predicted_class = np.argmax(node.value)
            elif self.metric_name == 'error':
                predicted_class = np.argmin(node.value)
            print(f"{indent}Leaf: predicted class = {predicted_class}")
            print(f"{indent}Leaf: class distribution = {node.value}")
        else:
            # Internal node: print its splitting rule
            if node.is_categorical:
                print(f"{indent}Node: [Feature {node.feature_index} == {node.split_value}]")
            else:
                print(f"{indent}Node: [Feature {node.feature_index} <= {node.split_value}]")

            # Recursively print the left and right subtrees
            print(f"{indent}  -> Left:")
            self.print_tree(node.left, depth + 2)
            print(f"{indent}  -> Right:")
            self.print_tree(node.right, depth + 2)

    def _assign_samples_to_leaves(self, X_new, Y_new):
        """
        Traverse the existing tree, placing each new (x, y) into the appropriate leaf node.
        """
        for x, y in zip(X_new, Y_new):
            node = self.root
            while True:
                # If it's a leaf, break
                if node.left is None and node.right is None:
                    # We've reached a leaf
                    break
                # Otherwise, decide if we go left or right
                if node.is_categorical:
                    if x[node.feature_index] == node.split_value:
                        node = node.left
                    else:
                        node = node.right
                else:
                    if x[node.feature_index] <= node.split_value:
                        node = node.left
                    else:
                        node = node.right
            # Now node is a leaf
            if node.X_leaf is None:
                node.X_leaf = []
                node.Y_leaf = []
            node.X_leaf.append(x)
            node.Y_leaf.append(y)
            node.value = np.array(node.Y_leaf).mean(axis=0)
    
    def _attempt_split_leaves(self, node, depth=0, max_depth=None):
        """
        Recursively check leaves; if they have enough new data,
        attempt to split them. If a good split is found, turn
        the leaf into an internal node with children.
        """
        if node is None:
            return
        
        # Only do this if the node is currently a leaf
        if node.left is None and node.right is None:
            # If we have stored data in this leaf:
            if node.X_leaf and len(node.X_leaf) >= self.min_samples_split:
                X_leaf = np.array(node.X_leaf)
                Y_leaf = np.array(node.Y_leaf)
                
                # Check stopping criteria, e.g., max_depth or gini ~ 0
                if depth < max_depth and self.metric(Y_leaf) > 1e-9:
                    # Attempt find best split
                    feature_index, split_value, is_categorical = self._find_best_split(X_leaf, Y_leaf)
                    
                    if feature_index is not None:
                        # We can do a split
                        node.feature_index = feature_index
                        node.split_value = split_value
                        node.is_categorical = is_categorical
                        node.value = None  # No longer a leaf distribution
                        
                        # Split data
                        if is_categorical:
                            X_left, Y_left, X_right, Y_right = self._split_dataset_categorical(
                                X_leaf, Y_leaf, feature_index, split_value
                            )
                        else:
                            X_left, Y_left, X_right, Y_right = self._split_dataset_numeric(
                                X_leaf, Y_leaf, feature_index, split_value
                            )
                        
                        # # Create children
                        # node.left = TreeNode()
                        # node.left.X_leaf = X_left.tolist()
                        # node.left.Y_leaf = Y_left.tolist()

                        # node.right = TreeNode()
                        # node.right.X_leaf = X_right.tolist()
                        # node.right.Y_leaf = Y_right.tolist()
                        
                        # # Clear data from this node
                        # node.X_leaf = None
                        # node.Y_leaf = None

                        # Recursively build subtrees
                        left_child = self._build_tree(X_left, Y_left, depth + 1)
                        right_child = self._build_tree(X_right, Y_right, depth + 1)

                        node.left = left_child
                        node.right = right_child
                        
                        # # Create the internal node
                        # return TreeNode(feature_index=feature_index,
                        #                 split_value=split_value,
                        #                 left=left_child,
                        #                 right=right_child,
                        #                 is_categorical=is_categorical)
        
        else:
            # Recurse for internal nodes
            self._attempt_split_leaves(node.left, depth + 1, max_depth)
            self._attempt_split_leaves(node.right, depth + 1, max_depth)

    def continue_fit(self, X_new, Y_new):
        """
        Incrementally fit the tree with new data (X_new, Y_new).
        """
        # 1) Assign new samples to existing leaves
        self._assign_samples_to_leaves(X_new, Y_new)

        # 2) Attempt splitting on those leaves
        current_depth = 0  # or compute if you want to track how deep the tree is
        self._attempt_split_leaves(self.root, depth=current_depth, max_depth=self.max_depth)

    def get_predictions_at_leaves(self):
        """
        Get a set of all possible predictions
        """

        predictions = []
        def traverse(node):
            if node is None:
                return
            if node.left is None and node.right is None:
                if self.metric_name == 'gini':
                    predictions.append(np.argmax(node.value))
                elif self.metric_name == 'error':
                    predictions.append(np.argmin(node.value))
            else:
                traverse(node.left)
                traverse(node.right)
        traverse(self.root)
        return list(set(predictions))
    
    def get_updated_feature_ranges(self, X_ranges):
        """
        Get the updated feature ranges for the new tree.
        """
        all_ranges = {}

        def update_ranges(predicted_class, ranges):
            if predicted_class not in all_ranges:
                all_ranges[predicted_class] = ranges
            else:
                for i, r in enumerate(ranges):
                    if i in self.categorical_features:
                        all_ranges[predicted_class][i] = all_ranges[predicted_class][i].union(r)
                    else:
                        all_ranges[predicted_class][i] = (min(all_ranges[predicted_class][i][0], r[0]), max(all_ranges[predicted_class][i][1], r[1]))

        def split_ranges(ranges, feature_index, split_value, is_categorical):
            feature_range = copy(ranges[feature_index])
            left_ranges = copy(ranges)
            right_ranges = copy(ranges)
            if is_categorical:
                left_range = {split_value}
                right_range = feature_range - left_range
            else:
                left_range = (feature_range[0], split_value)
                right_range = (split_value, feature_range[1])
            
            left_ranges[feature_index] = left_range
            right_ranges[feature_index] = right_range

            return left_ranges, right_ranges

        def traverse(node, ranges):
            if node is None:
                return
            if node.left is None and node.right is None:
                if self.metric_name == 'gini':
                    predicted_class = np.argmax(node.value)
                elif self.metric_name == 'error':
                    predicted_class = np.argmin(node.value)

                update_ranges(predicted_class, ranges)
            else:
                left_ranges, right_ranges = split_ranges(ranges, node.feature_index, node.split_value, node.is_categorical)
                traverse(node.left, left_ranges)
                traverse(node.right, right_ranges)
        traverse(self.root, X_ranges)
        return all_ranges









# --- Example Usage ---
if __name__ == "__main__":
    # Dummy dataset:
    # Let's say we have 4 samples, 2 features.
    # Feature 0 is numeric, Feature 1 is categorical (3 possible categories).
    X = np.array([
        [2.5, 1],  # sample 0
        [3.1, 2],  # sample 1
        [1.8, 1],  # sample 2
        [2.0, 0],  # sample 3
    ])

    # Labels: 2 classes (0 or 1) -> one-hot encoded
    Y_labels = np.array([0, 1, 0, 1])


    X = np.random.rand(100, 4)

    # Add categorical feature
    X[:, 1] = np.random.randint(0, 3, size=100)

    Y_labels = np.random.randint(0, 2, size=100)
    n_classes = len(np.unique(Y_labels))
    Y = np.eye(n_classes)[Y_labels]  # shape: (4, 2)

    Y = np.random.rand(100, 4)

    # Apply softmax to get probabilities for each row
    Y = np.exp(Y) / np.exp(Y).sum(axis=1, keepdims=True)

    # Suppose feature 1 is categorical
    categorical_features = [1]

    # Train the tree
    clf = DecisionTreeClassifier(max_depth=3,
                                 min_samples_split=1,
                                 categorical_features=categorical_features)
    clf.fit(X, Y)

    # Predictions
    preds = clf.predict(X)
    print("Predicted labels:", preds)
    print("True labels:     ", Y_labels)

    clf.print_tree()
