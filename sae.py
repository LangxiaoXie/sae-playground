"""A sparse autoencoder, small enough to read in one sitting.

The whole idea: a model's activation vector is a dense, entangled mix of many
underlying "features". Widen it into a much larger space where only a handful
of coordinates are allowed to be non-zero at once, and those coordinates tend
to line up with the features themselves.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class SAE(nn.Module):
    """Encoder: x -> sparse latents. Decoder: latents -> reconstruction of x.

    d_in      width of the activation being explained (5 for the toy, 768 for GPT-2)
    n_latents how many features we allow the SAE to look for; must exceed d_in
    """

    def __init__(
        self,
        d_in: int,
        n_latents: int,
        k: int | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.d_in = d_in
        self.n_latents = n_latents
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_in, n_latents))
        nn.init.kaiming_uniform_(self.W_enc)
        self.b_enc = nn.Parameter(torch.zeros(n_latents))

        self.W_dec = nn.Parameter(self.W_enc.detach().clone().T.contiguous())
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        if self.k is None:
            return z
        # TopK: keep only the k strongest latents, zero the rest. Sparsity is
        # then exact and needs no tuning -- but you no longer get to watch the
        # reconstruction/sparsity trade-off move, because you pinned one side.
        kept = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter_(-1, kept.indices, kept.values)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Force every latent's decoder direction to unit length.

        Without this the L1 penalty is gameable: the SAE can halve every latent
        activation and double the decoder norm, paying less penalty while
        reconstructing exactly as well. Sparsity would look better than it is.
        """
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def train_sae(
    sae: SAE,
    X: torch.Tensor,
    l1_coeff: float,
    steps: int = 2000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int | None = None,
) -> dict[str, list[float]]:
    """Fit an SAE to a fixed tensor of activations.

    The loss has exactly two terms, and the tension between them is the whole
    subject: squared reconstruction error wants many latents active, the L1
    penalty on the latents wants few. `l1_coeff` sets the exchange rate.
    """
    if seed is not None:
        torch.manual_seed(seed)

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.normalize_decoder_()
    history: dict[str, list[float]] = {"loss": [], "mse": [], "l0": []}

    # The activation cache lives on the CPU in fp16 (768 MB for stage 2); only the
    # sampled batch is moved and widened, so the GPU never holds the whole thing.
    param = next(sae.parameters())

    for _ in range(steps):
        index = torch.randint(0, X.shape[0], (batch_size,), device=X.device)
        batch = X[index].to(device=param.device, dtype=param.dtype)

        x_hat, z = sae(batch)
        mse = ((x_hat - batch) ** 2).sum(dim=-1).mean()
        l1 = z.abs().sum(dim=-1).mean()
        loss = mse + l1_coeff * l1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder_()

        history["loss"].append(loss.item())
        history["mse"].append(mse.item())
        history["l0"].append((z > 0).float().sum(dim=-1).mean().item())

    return history


@dataclass
class Recovery:
    """How well the SAE's learned directions match a known answer key.

    cos_matrix: (n_true_features, n_latents) cosine between every true feature
                and every learned decoder direction
    best_cos:   (n_true_features,) the best match found for each true feature
    best_latent:(n_true_features,) which latent that was
    """

    cos_matrix: np.ndarray
    best_cos: np.ndarray
    best_latent: np.ndarray

    def n_recovered(self, threshold: float = 0.9) -> int:
        return int((self.best_cos > threshold).sum())


def feature_recovery(sae: SAE, true_features: np.ndarray) -> Recovery:
    """Match each planted feature to its closest learned decoder direction."""
    with torch.no_grad():
        directions = sae.W_dec.detach().cpu().numpy()
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    truth = true_features / np.linalg.norm(true_features, axis=1, keepdims=True)

    cos_matrix = truth @ directions.T
    best_latent = cos_matrix.argmax(axis=1)
    best_cos = cos_matrix.max(axis=1)
    return Recovery(cos_matrix=cos_matrix, best_cos=best_cos, best_latent=best_latent)


@dataclass
class Metrics:
    """The three numbers that summarise an SAE.

    l0   average latents active per input -- how sparse the code actually is
    fvu  fraction of variance unexplained -- how much is lost reconstructing
    n_dead  latents that never fire at all; wasted capacity
    """

    l0: float
    fraction_variance_unexplained: float
    n_dead: int


@torch.no_grad()
def sae_metrics(sae: SAE, X: torch.Tensor, batch_size: int = 8192) -> Metrics:
    """Evaluate an SAE. Low l0 and low fvu together is the goal; either alone is easy."""
    total_sq_err = 0.0
    l0_sum = 0.0
    ever_fired = torch.zeros(sae.n_latents, dtype=torch.bool)

    for start in range(0, X.shape[0], batch_size):
        batch = X[start : start + batch_size]
        x_hat, z = sae(batch)
        total_sq_err += ((x_hat - batch) ** 2).sum().item()
        l0_sum += (z > 0).float().sum().item()
        ever_fired |= (z > 0).any(dim=0).cpu()

    # Compared against predicting the dataset mean, so 1.0 means "no better
    # than a constant" and 0.0 means perfect reconstruction.
    total_variance = ((X - X.mean(dim=0)) ** 2).sum().item()

    return Metrics(
        l0=l0_sum / X.shape[0],
        fraction_variance_unexplained=total_sq_err / total_variance,
        n_dead=int((~ever_fired).sum()),
    )
