# Locally Weighted Median Regressor

A simple regressor intended for estimating the conditional median fitted on observations of two one-dimensional variables. Compatible with the scikit-learn `Estimator` interface.

## Usage Examples

```Python
# Basic usage

from lowmer import LocallyWeightedMedianRegressor
import numpy as np

n = 200
X = np.random.normal(size=n)
y = 0.5 * X + np.random.lognormal(size=n)
X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
reg = LocallyWeightedMedianRegressor().fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

```Python
# Determining the optimal bandwidth by grid-search cross validation

from lowmer import LocallyWeightedMedianRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV

n = 200
X = np.random.normal(size=n)
y = 0.5 * X + np.random.lognormal(size=n)
X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

parameters = {"bandwidth": [0.25, 0.5, 1.0, 2.0, 5.0]}
model = LocallyWeightedMedianRegressor()
reg = GridSearchCV(model, parameters, scoring="neg_mean_absolute_error")
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

## Mathematical Details

Define the weighted median of a sorted sample $y_1, y_2, \ldots, y_n$ with positive weights $w_1, w_2, \ldots, w_n$ to be
$$
WM(y_1, y_2, \ldots, y_n; w_1, w_2, \ldots, w_n) = y_{\arg\min_j \sum_{i=1}^{j} w_i \ge \frac{\sum_{i=1}^{n} w_i}{2}}
$$
In other words, the weighted median is the value of the observation $y_j$ the cumulative weight up to whose index first exceeds half of the total weight.

For a new observation $x$, we calculate the weight associated with an observation in the training set $x_i$ as
$$
w_i(x) = \exp[-(\frac{x-x_i}{\theta})^2]
$$
where $\theta > 0$ is a tunable hyperparameter.

The LOWMER fitted with $x_1, x_2, \ldots, x_{n_{\text{train}}}$, $y_1, y_2, \ldots, y_{n_{\text{train}}}$ predicts the conditional median given $X=x$ to be
$$
M(x) = WM(y_1, y_2, \ldots, y_{n_{\text{train}}}; w_1(x), w_2(x), \ldots, w_{n_{\text{train}}}(x))
$$

In practice, the optimal $\theta$ can be obtained by grid-search cross validation on the training set.
