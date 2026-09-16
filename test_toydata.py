import numpy as np
import pytest

from toydata import make_superposition_data


def test_returns_samples_with_requested_shape():
    data = make_superposition_data(n_samples=100, n_true_features=8, n_dims=5, seed=0)
    assert data.X.shape == (100, 5)
    assert data.true_features.shape == (8, 5)
    assert data.coeffs.shape == (100, 8)


def test_true_features_are_unit_vectors():
    data = make_superposition_data(n_samples=10, n_true_features=8, n_dims=5, seed=0)
    norms = np.linalg.norm(data.true_features, axis=1)
    assert np.allclose(norms, 1.0)


def test_features_fire_at_the_requested_sparsity():
    data = make_superposition_data(
        n_samples=20000, n_true_features=8, n_dims=5, sparsity=0.1, seed=0
    )
    fraction_active = (data.coeffs > 0).mean()
    assert fraction_active == pytest.approx(0.1, abs=0.01)
