import numpy as np


def _check_inputs(X, y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {X.shape}")
    if y is None:
        return X
    y = np.asarray(y)
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0/1 labels")
    return X, y


class BipolarHebbianClassifier:
    """Single-pass Hebbian rule on bipolar (-1/+1) targets: w += x*y, b += y."""

    def __init__(self, input_dim):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def fit(self, X, y):
        X, y = _check_inputs(X, y)
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Expected {self.weights.shape[0]} features, got {X.shape[1]}")
        y_bipolar = np.where(y == 0, -1, 1)
        self.weights = self.weights + X.T @ y_bipolar
        self.bias = self.bias + float(y_bipolar.sum())
        return self

    def predict_score(self, X):
        return _check_inputs(X) @ self.weights + self.bias

    def predict(self, X):
        return np.where(self.predict_score(X) >= 0, 1, 0)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
