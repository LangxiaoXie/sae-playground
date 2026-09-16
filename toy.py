#!/usr/bin/env python3
"""Stage 1 in the terminal: a sparse autoencoder graded against a known answer key.

    python toy.py              # the whole walkthrough (~2 min)
    python toy.py --step       # pause between sections
    python toy.py -s 6         # jump straight to section 6
    python toy.py --steps 4000 # under-train it and watch recovery collapse

Same content as 01_toy.ipynb, rendered for a terminal instead of a browser.
All the real logic lives in sae.py / toydata.py, which are covered by the tests.
"""

import argparse

import numpy as np
import torch

from sae import SAE, feature_recovery, sae_metrics, train_sae
from termplot import barplot, heatmap, scatter
from toydata import make_superposition_data

B, D, R = "\033[1m", "\033[2m", "\033[0m"
SECTIONS = []


def section(title):
    def wrap(fn):
        SECTIONS.append((title, fn))
        return fn
    return wrap


def rule(title, n):
    print(f"\n{B}{'─' * 66}{R}")
    print(f"{B}  {n}. {title}{R}")
    print(f"{B}{'─' * 66}{R}")


def para(text):
    print("\n" + "\n".join("  " + line for line in text.strip().split("\n")))


def ensure_data(cfg, state):
    """Build the toy dataset once; sections that need it call this."""
    if "data" not in state:
        state["data"] = make_superposition_data(
            50_000, cfg.n_true, cfg.n_dims, sparsity=cfg.sparsity, seed=0
        )
        state["X"] = torch.tensor(state["data"].X, dtype=torch.float32)
    return state["data"]


def ensure_sae(cfg, state, announce=True):
    """Train the headline SAE once, so `-s 6` works without running `-s 5` first."""
    ensure_data(cfg, state)
    if "sae" not in state:
        if announce:
            print(f"\n  {D}(training the SAE first — {cfg.steps:,} steps){R}", flush=True)
        sae = SAE(d_in=cfg.n_dims, n_latents=cfg.n_latents, seed=0)
        state["history"] = train_sae(sae, state["X"], l1_coeff=cfg.l1,
                                     steps=cfg.steps, batch_size=1024, seed=0)
        state["sae"] = sae
    return state["sae"]


# ---------------------------------------------------------------------------


@section("The setup: more features than dimensions")
def s1(cfg, state):
    para(f"""A model stores more concepts than it has dimensions, letting them share
space. That is {B}superposition{R}, and its cost is that no single dimension
means anything on its own.

Here we plant {cfg.n_true} features in {cfg.n_dims} dimensions so we can grade the SAE later.""")
    data = ensure_data(cfg, state)
    print(f"\n  activations X        {tuple(data.X.shape)}")
    print(f"  features per sample  {(data.coeffs > 0).sum(1).mean():.2f} on average")
    print(f"  {D}(features are rare — real LM features are rarer still){R}")


@section("Why the features must interfere")
def s2(cfg, state):
    data = ensure_data(cfg, state)
    overlap = np.abs(data.true_features @ data.true_features.T)
    np.fill_diagonal(overlap, 0)
    para(f"""{cfg.n_true} unit vectors cannot all be perpendicular in {cfg.n_dims} dimensions —
there is not room. So every feature overlaps others.""")
    print(f"\n  largest |cos| between two different features: {B}{overlap.max():.3f}{R}")
    print(f"  {D}0.0 would mean perfectly separate; this is the interference{R}")
    heatmap(overlap, row_labels=[f"feat {i}" for i in range(cfg.n_true)],
            col_label="feature", title="Ground-truth features overlap")


@section("The problem, concretely")
def s3(cfg, state):
    data = ensure_data(cfg, state)
    i = int(np.argmax((data.coeffs > 0).sum(1)))
    para("Here is one activation vector. Which features produced it?")
    print(f"\n  x                  {np.round(data.X[i], 3)}")
    print(f"  features that fired {np.where(data.coeffs[i] > 0)[0].tolist()}")
    para(f"""Nothing about those {cfg.n_dims} numbers announces the answer — and this is the
{B}easy{R} case, where we know exactly how many things to look for.""")


@section("PCA can't do it")
def s4(cfg, state):
    from sklearn.decomposition import PCA
    data = ensure_data(cfg, state)
    pca = PCA(n_components=cfg.n_dims).fit(data.X)
    comps = pca.components_ / np.linalg.norm(pca.components_, axis=1, keepdims=True)
    truth = data.true_features / np.linalg.norm(data.true_features, axis=1, keepdims=True)
    cos = np.abs(truth @ comps.T)
    best = cos.argmax(1)

    para("""The obvious move is to find the data's principal directions. Watch how it
fails — it is more specific than "PCA is bad".""")
    print(f"\n  best match per feature  {np.round(cos.max(1), 2)}")
    print(f"  matched at >0.9         {(cos.max(1) > 0.9).sum()}/{cfg.n_true}")
    print(f"  but using only          {B}{len(set(best.tolist()))} distinct components{R}: {best.tolist()}")
    para(f"""Several features collide onto the {B}same{R} component. PCA is limited to
orthogonal directions and to at most {cfg.n_dims} of them, so it cannot give each
feature its own coordinate. That collision is the real failure.""")
    heatmap(cos, row_labels=[f"feat {i}" for i in range(cfg.n_true)],
            col_label="PCA component", title="Features collide onto shared components",
            gamma=4)


@section("Train the SAE")
def s5(cfg, state):
    para(f"""Same data, different constraint: allow {cfg.n_latents} directions instead of {cfg.n_dims},
and require that few are active at once.

    loss = ‖x − x̂‖²      reconstruct well → wants many latents active
         + λ · ‖z‖₁       stay sparse      → wants few

λ sets the exchange rate. That one number is most of the tuning.""")
    ensure_data(cfg, state)
    print(f"\n  training {cfg.steps:,} steps at λ={cfg.l1} ...", flush=True)
    ensure_sae(cfg, state, announce=False)
    history = state["history"]
    tail = slice(-200, None)
    print(f"  final reconstruction error {np.mean(history['mse'][tail]):.5f}")
    print(f"  final L0                   {np.mean(history['l0'][tail]):.2f} latents active")


@section("The money plot")
def s6(cfg, state):
    sae = ensure_sae(cfg, state)
    rec = feature_recovery(sae, state["data"].true_features)
    m = sae_metrics(sae, state["X"])
    para(f"""Every true feature against every learned direction. Success looks like one
bright cell per row: that feature was found, and it lives in one latent.""")
    heatmap(rec.cos_matrix, row_labels=[f"feat {i}" for i in range(cfg.n_true)],
            col_label="SAE latent", title="One bright cell per row = recovered",
            highlight=rec.best_latent, gamma=4)
    ok = rec.n_recovered(0.9)
    colour = "\033[38;5;220m" if ok == cfg.n_true else "\033[38;5;203m"
    print(f"\n  recovered            {colour}{ok}/{cfg.n_true}{R} features at cos > 0.9")
    print(f"  best cosine each     {np.round(rec.best_cos, 3)}")
    print(f"  L0                   {m.l0:.2f} of {cfg.n_latents} active")
    print(f"  variance unexplained {m.fraction_variance_unexplained:.1%}")
    print(f"  dead latents         {m.n_dead}")


@section("The trade-off, drawn")
def s7(cfg, state):
    ensure_data(cfg, state)
    para("""λ does not have a correct value — it picks a point on a curve. Two things
behave differently, and the difference matters:""")
    lambdas = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    l0s, fvus, recs, deads = [], [], [], []
    print(f"\n  {'λ':>8} {'L0':>7} {'unexplained':>12} {'recovered':>10} {'dead':>6}")
    for lam in lambdas:
        s = SAE(d_in=cfg.n_dims, n_latents=cfg.n_latents, seed=0)
        train_sae(s, state["X"], l1_coeff=lam, steps=cfg.steps, batch_size=1024, seed=0)
        m, r = sae_metrics(s, state["X"]), feature_recovery(s, state["data"].true_features)
        l0s.append(m.l0); fvus.append(m.fraction_variance_unexplained * 100)
        recs.append(r.n_recovered(0.9)); deads.append(m.n_dead)
        print(f"  {lam:>8g} {m.l0:>7.2f} {m.fraction_variance_unexplained:>11.1%}"
              f" {r.n_recovered(0.9):>7}/{cfg.n_true} {m.n_dead:>6}", flush=True)

    scatter(l0s, fvus, [f"λ={l:g}" for l in lambdas],
            title="Sparser codes reconstruct worse",
            xlabel="L0 (lower = sparser)  →  variance unexplained %")
    barplot([f"λ={l:g}" for l in lambdas], [float(r) for r in recs],
            title=f"Recovery (of {cfg.n_true})", fmt="{:.0f}")
    barplot([f"λ={l:g}" for l in lambdas], [float(d) for d in deads],
            title="Dead latents", fmt="{:.0f}")
    para(f"""Error and dead latents climb {B}monotonically{R} with λ. Recovery does not —
it stays high across a wide band and wobbles, so tuning on recovery alone
would mislead you. At λ=0.3 you can still find every feature while having
thrown away a tenth of the variance and killed half the dictionary.""")


@section("TopK: the other way to be sparse")
def s8(cfg, state):
    ensure_data(cfg, state)
    para(f"""Instead of penalising density you can forbid it: keep the k largest latents,
zero the rest. Sparsity becomes exact and there is no λ to tune.

The catch is that you must already know how sparse the truth is. This data
fires {(state['data'].coeffs > 0).sum(1).mean():.1f} features per sample.""")
    ks, recs = [], []
    print(f"\n  {'k':>4} {'L0':>6} {'unexplained':>12} {'recovered':>10} {'dead':>6}")
    for k in [1, 2, 3, 5, 10]:
        s = SAE(d_in=cfg.n_dims, n_latents=cfg.n_latents, k=k, seed=0)
        train_sae(s, state["X"], l1_coeff=0.0, steps=6_000, batch_size=1024, seed=0)
        m, r = sae_metrics(s, state["X"]), feature_recovery(s, state["data"].true_features)
        ks.append(k); recs.append(float(r.n_recovered(0.9)))
        print(f"  {k:>4} {m.l0:>6.1f} {m.fraction_variance_unexplained:>11.1%}"
              f" {r.n_recovered(0.9):>7}/{cfg.n_true} {m.n_dead:>6}", flush=True)
    barplot([f"k={k}" for k in ks], recs, title=f"Recovery (of {cfg.n_true})", fmt="{:.0f}")
    para("""Recovery falls as k rises above the true sparsity: forced to fire five
latents when one real feature is present, the SAE splits features to fill
the quota. With L1 you never had to guess that number — the penalty found it.""")


@section("Things worth breaking")
def s9(cfg, state):
    para(f"""{B}python toy.py --steps 4000{R}
    Recovery drops to roughly 4/8 while reconstruction error barely moves.
    Under-training does not announce itself: the SAE looks healthy and is
    quietly merging two features into one latent. This is the most common
    way to be fooled by an SAE, and on a real model you cannot detect it.

{B}python toy.py --sparsity 0.2{R}
    Features now co-occur constantly, so the SAE learns latents for
    combinations. Ground truth stops being recoverable at all.

{B}python toy.py --latents 8{R}
    Exactly as many latents as features, no slack. Recovery collapses —
    overcompleteness is not a luxury, it is the room needed to separate things.

{B}python toy.py --dims 8{R}
    No superposition at all. Everything works trivially, which is the point:
    SAEs solve a problem that only exists when representations are crowded.

Then: {B}02_gpt2.ipynb{R} — same SAE class, real GPT-2, and no answer key.""")


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--l1", type=float, default=1e-2)
    p.add_argument("--sparsity", type=float, default=0.05)
    p.add_argument("--latents", dest="n_latents", type=int, default=32)
    p.add_argument("--dims", dest="n_dims", type=int, default=5)
    p.add_argument("--true", dest="n_true", type=int, default=8)
    p.add_argument("-s", "--section", type=int, help="run one section only")
    p.add_argument("--step", action="store_true", help="pause between sections")
    cfg = p.parse_args()

    print(f"\n{B}  A sparse autoencoder, with an answer key{R}")
    print(f"  {D}{cfg.n_true} features in {cfg.n_dims} dims · {cfg.n_latents} latents · "
          f"λ={cfg.l1} · {cfg.steps:,} steps · sparsity {cfg.sparsity}{R}")

    state = {}
    chosen = (list(range(len(SECTIONS))) if cfg.section is None
              else [cfg.section - 1])
    if not all(0 <= i < len(SECTIONS) for i in chosen):
        p.error(f"--section must be 1..{len(SECTIONS)}")

    for n, i in enumerate(chosen):
        title, fn = SECTIONS[i]
        rule(title, i + 1)
        fn(cfg, state)
        if cfg.step and n != len(chosen) - 1:
            input(f"\n  {D}[enter]{R} ")
    print()


if __name__ == "__main__":
    main()
