# SAE playground

A sparse autoencoder you can read in one sitting, first checked against data whose
features are known, then pointed at GPT-2.

## Run it

```bash
.venv/bin/python -m pytest          # 26 tests, ~20s
```

Then, for stage 1, either:

```bash
.venv/bin/python toy.py         # terminal walkthrough, ~2 min, no browser needed
```

or open **`01_toy.ipynb`** and pick the `playground (SAE)` kernel.

`toy.py` covers the same nine sections as the notebook and produces the same numbers.
Plots are drawn as coloured blocks straight into the terminal, since plotly needs a
browser. Useful flags:

```bash
.venv/bin/python toy.py --step         # pause between sections
.venv/bin/python toy.py -s 6           # jump to one section (trains what it needs)
.venv/bin/python toy.py --steps 4000   # under-train it; recovery collapses to ~4/8
.venv/bin/python toy.py --dims 8       # remove superposition entirely
```

## The two notebooks

**`01_toy.ipynb`** — synthetic activations built from 8 features planted in 5
dimensions. Because you know the answer, you can grade the SAE: it recovers 8/8
planted features at cosine > 0.9. Also shows why PCA can't do this, what the
sparsity/reconstruction trade-off actually looks like, and how TopK differs from an
L1 penalty. Runs in about two minutes, CPU only.

**`02_gpt2.ipynb`** — the same `SAE` class on GPT-2's layer-6 residual stream. Train
one, read its features by looking at the text that fires them, then load a production
SAE (Joseph Bloom's `gpt2-small-res-jb`) and compare. Ends with steering: push a
latent's direction into the forward pass and watch the output bend.

It uses TopK (`k=32`) rather than an L1 penalty. On real activations L1 needs
calibrating and the useful range is nowhere near notebook 1's — measured on this
cache, λ=2.0 gives L0≈593 where you want tens. TopK sets sparsity by construction.

**This notebook heats the machine.** The training cell runs the GPU flat out for
2–3 minutes (defaults: 4096 latents, 1500 steps). Everything else is cheap, and
`SKIP_TRAINING = True` at the top skips it entirely and uses only the pretrained SAE.

Notebook 2 needs an activation cache, built once:

```bash
.venv/bin/python build_cache.py --n 500000     # ~2 min, writes 733 MB
```

## The code

| file | what it is |
|---|---|
| `sae.py` | the SAE, its training loop, and metrics — ~150 lines, no framework |
| `toydata.py` | synthetic activations with a known answer key |
| `activations.py` | streams and caches GPT-2 activations |
| `featureviz.py` | maps a latent back to the text that fired it |
| `build_cache.py` | one-time cache builder for notebook 2 |
| `toy.py` | stage 1 as a terminal walkthrough |
| `termplot.py` | block-character heatmaps/plots for the terminal |
| `saeplay.py` | pre-existing: SAE latents as features for a downstream probe |

Tests mirror those files. The interesting one is
`test_sae.py::test_trained_sae_recovers_the_planted_features` — it asserts the SAE
finds the features that generated the data, which is only checkable because the toy
data has ground truth.

## Disk

Everything this creates is deletable:

```bash
rm acts_layer6*.pt sae_layer6_*.pt     # ~800 MB, rebuildable
rm -rf ~/.cache/huggingface            # ~760 MB: GPT-2, the corpus, the reference SAE
```

To keep model downloads off the boot volume entirely, set `HF_HOME` to another
location before running anything.

## Notes

- Runs on Apple Silicon via MPS. TransformerLens prints a generic warning that MPS
  "may produce silently incorrect results" — checked against CPU on this setup and
  activations match to cosine 1.000000, so the cache is sound.
- `sae_lens` ≥ 6 changed `SAE.from_pretrained` to return the SAE alone rather than a
  `(sae, cfg, sparsity)` tuple. `saeplay.py` predates that and still unpacks three
  values.
