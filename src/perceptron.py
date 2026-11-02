import numpy as np

try:
    from .hebbian import _check_inputs
except ImportError:  # legacy joblib pickles import this module as top-level `perceptron`
    from hebbian import _check_inputs


class CustomBipolarPerceptron:
    """Classic perceptron on bipolar targets with early stopping once an epoch has no errors."""

    def __init__(self, input_dim, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def fit(self, X, y):
        X, y = _check_inputs(X, y)
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Expected {self.weights.shape[0]} features, got {X.shape[1]}")
        y_bipolar = np.where(y == 0, -1, 1)
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y_bipolar):
                pred = 1 if np.dot(xi, self.weights) + self.bias >= 0 else -1
                if pred != yi:
                    self.weights += self.lr * yi * xi
                    self.bias += self.lr * yi
                    errors += 1
            if errors == 0:
                break
        return self

    def predict(self, X):
        return np.where(_check_inputs(X) @ self.weights + self.bias >= 0, 1, 0)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
