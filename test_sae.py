import numpy as np
import pytest
import torch

from sae import SAE


def test_encode_produces_one_nonnegative_value_per_latent():
    sae = SAE(d_in=5, n_latents=32, seed=0)
    x = torch.randn(100, 5)

    z = sae.encode(x)

    assert z.shape == (100, 32)
    assert (z >= 0).all(), "ReLU encoder must never emit a negative activation"


def test_decode_maps_latents_back_to_input_space():
    sae = SAE(d_in=5, n_latents=32, seed=0)
    z = torch.rand(100, 32)

    x_hat = sae.decode(z)

    assert x_hat.shape == (100, 5)


def test_normalize_decoder_makes_every_feature_direction_unit_length():
    sae = SAE(d_in=5, n_latents=32, seed=0)
    with torch.no_grad():
        sae.W_dec.mul_(7.0)

    sae.normalize_decoder_()

    norms = sae.W_dec.norm(dim=1)
    assert torch.allclose(norms, torch.ones(32), atol=1e-6), (
        "unnormalized decoder lets the SAE cheat the L1 penalty by shrinking z "
        "and growing the decoder to compensate"
    )


def test_training_reduces_reconstruction_error():
    from toydata import make_superposition_data
    from sae import train_sae

    data = make_superposition_data(n_samples=20000, seed=0)
    X = torch.tensor(data.X, dtype=torch.float32)
    sae = SAE(d_in=5, n_latents=32, seed=0)

    with torch.no_grad():
        before = ((sae(X)[0] - X) ** 2).mean().item()
    train_sae(sae, X, l1_coeff=1e-3, steps=500, batch_size=1024, seed=0)
    with torch.no_grad():
        after = ((sae(X)[0] - X) ** 2).mean().item()

    assert after < before / 10, f"error barely moved: {before:.5f} -> {after:.5f}"


def test_trained_sae_recovers_the_planted_features():
    """The claim SAEs make, checked against an answer key.

    Individual dimensions of X are meaningless mixtures because 8 features are
    crammed into 5 dimensions. If the SAE works, its decoder directions should
    line up with the features that actually generated the data.
    """
    from toydata import make_superposition_data
    from sae import train_sae, feature_recovery

    data = make_superposition_data(
        n_samples=50000, n_true_features=8, n_dims=5, sparsity=0.05, seed=0
    )
    X = torch.tensor(data.X, dtype=torch.float32)

    sae = SAE(d_in=5, n_latents=32, seed=0)
    train_sae(sae, X, l1_coeff=1e-2, steps=15000, batch_size=1024, seed=0)

    recovery = feature_recovery(sae, data.true_features)

    assert recovery.n_recovered(threshold=0.9) >= 7, (
        f"only {recovery.n_recovered(0.9)}/8 features recovered; "
        f"best cosines = {np.round(recovery.best_cos, 3)}"
    )


def test_topk_sae_activates_exactly_k_latents():
    """The other way to get sparsity: fix L0 by construction instead of penalising it.

    With `k` set there is no L1 term and nothing to tune -- you just declare how
    many features may fire at once.
    """
    sae = SAE(d_in=5, n_latents=32, k=3, seed=0)
    x = torch.randn(500, 5)

    z = sae.encode(x)

    n_active = (z != 0).sum(dim=-1)
    assert (n_active == 3).all(), f"expected exactly 3 active, saw {n_active.unique().tolist()}"


def test_l0_counts_the_average_number_of_active_latents():
    from sae import sae_metrics

    sae = SAE(d_in=5, n_latents=32, k=3, seed=0)
    X = torch.randn(200, 5)

    assert sae_metrics(sae, X).l0 == pytest.approx(3.0)


def test_variance_unexplained_is_one_when_the_sae_only_predicts_the_mean():
    """The baseline any SAE must beat: reconstructing every input as the dataset
    mean explains none of the variance."""
    from sae import sae_metrics

    sae = SAE(d_in=5, n_latents=32, seed=0)
    X = torch.randn(2000, 5)
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.zero_()
        sae.b_dec.copy_(X.mean(dim=0))

    assert sae_metrics(sae, X).fraction_variance_unexplained == pytest.approx(1.0, abs=1e-3)


def test_counts_latents_that_never_fire_on_any_input():
    from sae import sae_metrics

    sae = SAE(d_in=5, n_latents=32, seed=0)
    X = torch.randn(200, 5)
    with torch.no_grad():
        sae.b_enc[:10] = -1e9  # force exactly 10 latents permanently off

    assert sae_metrics(sae, X).n_dead == 10


def test_training_accepts_a_half_precision_cache():
    """Real activations are cached in fp16 on CPU while the SAE trains in fp32 on
    the GPU. train_sae must bridge that rather than making the caller do it."""
    from sae import train_sae

    X_fp16 = torch.randn(5000, 5, dtype=torch.float16)
    sae = SAE(d_in=5, n_latents=32, seed=0)

    history = train_sae(sae, X_fp16, l1_coeff=1e-2, steps=50, batch_size=256, seed=0)

    assert len(history["mse"]) == 50
    assert not np.isnan(history["mse"][-1])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_training_bridges_a_cpu_cache_to_a_gpu_model():
    """The stage-2 shape: a 733 MB fp16 cache stays in RAM while the SAE trains on
    the GPU. Without this the caller has to move every batch by hand."""
    from sae import train_sae

    X = torch.randn(5000, 5, dtype=torch.float16)  # CPU cache
    sae = SAE(d_in=5, n_latents=32, seed=0).to("mps")

    history = train_sae(sae, X, l1_coeff=1e-2, steps=20, batch_size=256, seed=0)

    assert len(history["mse"]) == 20
