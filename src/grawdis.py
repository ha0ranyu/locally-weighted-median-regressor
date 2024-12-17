import numpy as np
from scipy.stats import iqr


class Grawdis:
    """A model for calculating the robust and automatically weighted Gower's distance."""

    def __init__(self):
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, var_type: str):
        """Fits the model.

        Parameters
        ----------
        X_train : ndarray, shape (n_samples, n_features)
            The training input samples.

        var_type : str
            Variable types of components corresponding to columns in X_train.
            Must consist of letters c (continuous), u (unordered discrete) and o (ordered discrete) and have the same length as the number of columns in X_train.

        Returns
        -------
        self : object
            Returns self.
        """

        # Validate the input
        if not isinstance(X_train, np.ndarray):
            raise TypeError(
                f"X_train must be a numpy array. Got {type(X_train)} instead."
            )
        if not isinstance(var_type, str):
            raise TypeError(f"var_type must be a string. Got {type(var_type)} instead.")
        if set(var_type) - set("cuo"):
            raise ValueError(
                "The variable type must be one of c (continuous), u (unordered discrete) or o (ordered discrete)."
            )
        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be a 2-dimensional array. Got {X_train.dim} instead."
            )
        self.n_obs, self.n_features = X_train.shape
        if len(var_type) != self.n_features:
            raise ValueError(
                "The length of var_type must be the same as the number of columns in X_train."
            )

        # Initialize some variables
        self.X_train = X_train
        self.var_type = var_type
        self.scale_factors = [None for _ in range(self.n_features)]
        component_distance_matrices = np.empty(
            (self.n_features, int(self.n_obs * (self.n_obs - 1) / 2))
        )

        # Iterate over each feature
        for feature_index in range(self.n_features):
            feature_data = X_train[:, feature_index]
            feature_type = var_type[feature_index]
            # Store the scale factors
            if feature_type == "c":
                self.scale_factors[feature_index] = iqr(feature_data)
            elif feature_type == "o":
                self.scale_factors[feature_index] = (
                    feature_data.max() - feature_data.min()
                )
            # Obtain the distance matrix of the current component and extract the upper-triangular part
            component_distance_matrices[feature_index] = (
                Grawdis._get_component_distance_matrix(
                    feature_data,
                    feature_data,
                    feature_type,
                    self.scale_factors[feature_index],
                )[np.triu_indices(self.n_obs, 1)]
            )
        # Optimize the component weights
        if self.n_features == 1:
            self.component_weights = np.ndarray([1])
        else:
            self.component_weights = self._get_component_weights(
                component_distance_matrices
            )
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calculate the weighted Gower's distances between observations in X and self.X_train_.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            An array containing samples with the same features as self.X_train_.

        Returns
        -------
        distances : ndarray, shape (n_samples, self.X_train_.shape[1])
            A 2-dimensional array where distances[i, j] is the weighted Gower's distance between the ith observation of X and the jth observation of self.X_train_
        """

        # Validate the input
        if not self.is_fitted:
            raise AttributeError("The instance has not been fitted.")
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array. Got {type(X)} instead.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-dimensional array. Got {X.ndim} instead.")
        if X.shape[1] != self.n_features:
            raise ValueError(
                "X must contain the same number of columns as in the training set."
            )

        nrows = X.shape[0]
        # Calculate the distances componentwise
        component_distances = np.empty((self.n_features, nrows, self.n_obs))
        for component in range(self.n_features):
            component_distances[component] = self._get_component_distance_matrix(
                X[:, component],
                self.X_train[:, component],
                self.var_type[component],
                self.scale_factors[component],
            )
        # Calculate the overall distance
        distances = np.sum(
            self.component_weights.reshape(-1, 1, 1) * component_distances, axis=0
        )
        return distances

    @staticmethod
    def _get_component_distance_matrix(
        data_1: np.ndarray, data_2: np.ndarray, type: str, scale_factor: float | None
    ) -> np.ndarray:
        # Decide the formula used for calculating the distances by variable type
        if type == "u":
            component_distance_matrix = data_1.reshape(-1, 1) == data_2.reshape(1, -1)
        else:
            component_distance_matrix = np.minimum(
                (np.abs(data_1.reshape(-1, 1) - data_2.reshape(1, -1)) / scale_factor),
                1,
            )
        return component_distance_matrix

    def _get_component_weights(self, distance_matrices) -> np.ndarray:
        # Solve for the optimal weights
        sigma = np.sqrt(
            np.var(distance_matrices, axis=1) * self.n_obs * (self.n_obs - 1) / 2
        )
        A = np.corrcoef(distance_matrices)
        A -= A[0, :]
        A *= sigma
        A[0, :] = 1
        b = np.zeros((self.n_features,))
        b[0] = 1
        weights = np.linalg.solve(A, b)
        return weights
