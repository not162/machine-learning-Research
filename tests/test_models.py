import numpy as np
import pytest

from src import BipolarHebbianClassifier, CustomBipolarPerceptron


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (30, 4)), rng.normal(2, 0.5, (30, 4))])
    y = np.array([0] * 30 + [1] * 30)
    return X, y


def test_hebbian_separable(blobs):
    X, y = blobs
    assert BipolarHebbianClassifier(4).fit(X, y).score(X, y) > 0.95


def test_perceptron_separable(blobs):
    X, y = blobs
    assert CustomBipolarPerceptron(4, lr=0.1, epochs=50).fit(X, y).score(X, y) == 1.0


def test_rejects_bad_labels_and_shapes(blobs):
    X, y = blobs
    with pytest.raises(ValueError):
        BipolarHebbianClassifier(4).fit(X, y + 1)
    with pytest.raises(ValueError):
        CustomBipolarPerceptron(3).fit(X, y)
