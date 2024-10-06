import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils.validation import check_is_fitted


class LocallyWeightedMedianRegressor(BaseEstimator, RegressorMixin):
    """A regressor for estimating the estimated conditional median.

    Implements the locally weighted median regression algorithm for one-dimensional inputs and outputs.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Controls the smoothness of the kernel function in the Gaussian kernel regressor. It determines the width of the kernel, affecting how much neighboring points influence the prediction. A smaller bandwidth leads to a more localized fit, while a larger bandwidth results in a smoother, more generalized fit.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from lowmer import LocallyWeightedMedianRegressor
    >>> import numpy as np
    >>> n = 200
    >>> X = np.random.normal(size=n)
    >>> y = 0.5 * X + np.random.lognormal(size=n)
    >>> X_train, X_test = X[:100], X[100:]
    >>> y_train, y_test = y[:100], y[100:]
    >>> X_train = X_train.reshape(-1, 1)
    >>> X_test = X_test.reshape(-1, 1)
    >>> reg = LocallyWeightedMedianRegressor().fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "bandwidth": [float],
    }

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fits the estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features=1)
            The one-column training input samples.

        y : array-like, shape (n_samples,)
            The observed target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        if self.bandwidth < 0:
            raise ValueError("The bandwidth must be positive.")
        X, y = self._validate_data(X, y, accept_sparse=False)
        if X.shape[1] != 1:
            raise ValueError("X must contain exactly one column.")

        # Sort values by y in ascending order.
        sorted_indices = np.argsort(y)
        self.X_ = X[sorted_indices]
        self.y_ = y[sorted_indices]
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """Predict the conditional medians using local weighted median regression.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features=1)
            A one-column array containing input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The estimated conditional medians.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=False, reset=False)
        weights = self._get_weights(self.X_, X, self.bandwidth)
        # NOTE: Do not use np.percentile since it attempts to sort the input array.
        medians = self._weighted_median(self.y_, weights)
        return medians

    @staticmethod
    def _get_weights(X_train, X_test, bandwidth):
        # Calculate the weights using the Gaussian kernel function
        return np.exp(-(((X_test - X_train.reshape(1, -1)) / bandwidth) ** 2))

    @staticmethod
    def _weighted_median(values, weights):
        # Compute cumulative weights
        cumulative_weights = np.cumsum(weights, axis=1)
        # Find the median
        total_weights = cumulative_weights[:, -1]
        # Assume the values are already sorted in ascending order
        median_index = [
            np.searchsorted(cumulative_weight, total_weight / 2.0)
            for cumulative_weight, total_weight in zip(
                cumulative_weights, total_weights
            )
        ]
        return values[median_index]
