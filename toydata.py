"""Synthetic activations with known ground-truth features.

The point of this module is that it gives you an answer key. Real model
activations have no labelled "true features", so you can never fully verify
that an SAE found the right thing. Here you plant the features yourself.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SuperpositionData:
    """Synthetic activations plus the ground truth that generated them.

    X:             (n_samples, n_dims)          the "activations" an SAE sees
    true_features: (n_true_features, n_dims)    unit vectors, the answer key
    coeffs:        (n_samples, n_true_features) how strongly each fired; mostly 0
    """

    X: np.ndarray
    true_features: np.ndarray
    coeffs: np.ndarray


def make_superposition_data(
    n_samples: int,
    n_true_features: int = 8,
    n_dims: int = 5,
    sparsity: float = 0.05,
    seed: int = 0,
) -> SuperpositionData:
    """Plant `n_true_features` directions in `n_dims` dimensions and fire them sparsely.

    When n_true_features > n_dims the directions cannot all be orthogonal, so
    they interfere: that crowding is superposition, and it is what makes
    individual dimensions of X uninterpretable.
    """
    rng = np.random.default_rng(seed)

    true_features = rng.normal(size=(n_true_features, n_dims))
    true_features /= np.linalg.norm(true_features, axis=1, keepdims=True)

    active = rng.random((n_samples, n_true_features)) < sparsity
    magnitudes = rng.uniform(0.0, 1.0, size=(n_samples, n_true_features))
    coeffs = active * magnitudes

    X = coeffs @ true_features

    return SuperpositionData(X=X, true_features=true_features, coeffs=coeffs)
