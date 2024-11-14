# Locally Weighted Median Regressor

A simple regressor intended for estimating the conditional median of a random variable $Y$ given a random vector $\bm{X}$. Compatible with the scikit-learn `Estimator` interface.

## Usage Examples

```Python
# Basic usage

import numpy as np
from lowmer import LocallyWeightedMedianRegressor

n = 200
X = np.random.normal(size=(n, 2))
y = 2 * X[:, 0] - X[:, 1]**2 + np.random.lognormal(size=n)
X_train, X_test = X[:100, :], X[100:, :]
y_train, y_test = y[:100], y[100:]
reg = LocallyWeightedMedianRegressor(bandwidth=0.5).fit(X_train, y_train, var_type="cc")
y_pred = reg.predict(X_test)
```

```Python
# Determining the optimal bandwidth by grid-search cross validation

import numpy as np
from sklearn.model_selection import GridSearchCV
from lowmer import LocallyWeightedMedianRegressor

n = 200
X = np.random.normal(size=(n, 2))
y = 2 * X[:, 0] - X[:, 1]**2 + np.random.lognormal(size=n)
X_train, X_test = X[:100, :], X[100:, :]
y_train, y_test = y[:100], y[100:]

parameters = {"bandwidth": [0.05, 0.1, 0.25, 0.5, 1.0]}
model = LocallyWeightedMedianRegressor()
reg = GridSearchCV(model, parameters, scoring="neg_mean_absolute_error")
reg.fit(X_train, y_train, var_type="cc")
y_pred = reg.predict(X_test)
```

## Mathematical Details

The weighted median of a sorted sample $y_1, y_2, \ldots, y_n$ with positive weights $w_1, w_2, \ldots, w_n$ is defined to be
```math
WM(y_1, y_2, \ldots, y_n; w_1, w_2, \ldots, w_n) = y_{\arg\min_j \sum_{i=1}^{j} w_i \ge \frac{\sum_{i=1}^{n} w_i}{2}}
```
In other words, the weighted median is the value of the observation $y_j$ the cumulative weight up to whose index first exceeds half of the total weight.

For a new observation $\mathbf{x}$, we calculate the weight associated with an observation in the training set $\mathbf{x}_i$ as
```math
w_i(\mathbf{x}) = \exp\left\{-\left[\frac{d(\mathbf{x}-\mathbf{x}_i)}{\theta}\right]^2\right\}
```
where $\theta > 0$ is a tunable hyperparameter and $d(\cdot)$ is the robust and automatically weighted Gower's distance.

The component-wise distance between two observations $\mathbf{x}_i, \mathbf{x}_j$ is calculated as
```math
d_k(\mathbf{x}_i, \mathbf{x}_j) = \begin{cases}
    \mathbf{1}(\mathbf{x}_{ik} = \mathbf{x}_{jk}), & \text{The $k$th component is nominal}\\
    \min\left\{1, \frac{|\mathbf{x}_{ik} - \mathbf{x}_{jk}|}{Range_k}\right\}, & \text{The $k$th component is ordered discrete}\\
    \min\left\{1, \frac{|\mathbf{x}_{ik} - \mathbf{x}_{jk}|}{IQR_k}\right\}, & \text{The $k$th component is continuous}\\
\end{cases}
```
where the range and interquartile range are estimated from the training set.

The overall distance between two observations $\mathbf{x}_i, \mathbf{x}_j$ is calculated as
```math
d(\mathbf{x}_i, \mathbf{x}_j) = \sum_k w_k d_k(\mathbf{x}_i, \mathbf{x}_j)
```
where the weights are chosen such that the differences among the correlations between each single component-wise distance and the overall distance are minimized.
In the absence of missing values, an analytical solution that involves solving a system of linear equations exists.

The LOWMER fitted with $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_{n_{\text{train}}}$, $y_1, y_2, \ldots, y_{n_{\text{train}}}$ predicts the conditional median given $\mathbf{X}=\mathbf{x}$ to be
```math
M(x) = WM[y_1, y_2, \ldots, y_{n_{\text{train}}}; w_1(\mathbf{x}), w_2(\mathbf{x}), \ldots, w_{n_{\text{train}}}(\mathbf{x})]
```

In practice, the optimal $\theta$ can be obtained by grid-search cross validation on the training set.

## License

This project is released under the MIT License.


## References

- Marcello D'Orazio (2021). "Distances with mixed type variables some modified Gower's coefficients": https://doi.org/10.48550/arXiv.2101.02481

- Francesco de Bello, Zoltán Botta-Dukát, Jan Lepš, Pavel Fibich (2021), "Towards a more balanced combination of multiple traits when computing functional differences between species": https://doi.org/10.1111/2041-210X.13537