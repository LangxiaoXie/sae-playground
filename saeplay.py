"""
SAE vs. Linear Probe: a minimal downstream-task comparison.

Mirrors, in miniature, the GDM mech interp team's OOD probing setup:
https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/

Concept probed: sentiment (positive vs negative), on GPT-2 small.
- Dense linear probe on raw residual stream activations.
- SAE probes: best single latent, k-sparse (top-k by mean diff), and a
  probe trained on the SAE reconstruction instead of the raw activation.

Train/val use one family of sentence templates; a held-out OOD set uses a
different phrasing style, so you can see whether each probe generalizes
or was just fitting spurious correlations in the training templates.

Install:
    pip install transformer_lens sae_lens scikit-learn torch --break-system-packages

Needs HuggingFace access to pull GPT-2 small + a pretrained SAE
(Joseph Bloom's "gpt2-small-res-jb" release). Run locally, not sandboxed.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from transformer_lens import HookedTransformer
from sae_lens import SAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 6  # probe at resid_pre of this layer; middling depth, decent semantic content

# ---------------------------------------------------------------------------
# 1. Data: template-based sentiment sentences.
#    ID templates for train/val, a distinct OOD template family for testing
#    generalization (mirrors the AF post's train-on-some-jailbreaks,
#    test-on-others design).
# ---------------------------------------------------------------------------

NOUNS = ["movie", "meal", "hotel", "book", "phone", "car", "concert", "restaurant",
         "laptop", "vacation", "class", "game", "song", "painting", "apartment"]

ID_POS_TEMPLATES = [
    "I love this {noun}.",
    "This {noun} is amazing.",
    "What a wonderful {noun}!",
    "The {noun} was fantastic.",
    "I really enjoyed the {noun}.",
]
ID_NEG_TEMPLATES = [
    "I hate this {noun}.",
    "This {noun} is terrible.",
    "What an awful {noun}!",
    "The {noun} was horrible.",
    "I really disliked the {noun}.",
]

# Different syntax/register entirely -- tests whether the probe learned
# "positive/negative sentiment" or just surface patterns from the ID templates.
OOD_POS_TEMPLATES = [
    "Honestly, the {noun} exceeded every expectation I had.",
    "Nothing about the {noun} disappointed me in the slightest.",
    "They really outdid themselves with this {noun}.",
    "I'd recommend the {noun} to anyone without hesitation.",
]
OOD_NEG_TEMPLATES = [
    "Honestly, the {noun} fell short of every expectation I had.",
    "Nothing about the {noun} impressed me in the slightest.",
    "They really let me down with this {noun}.",
    "I'd warn anyone away from the {noun} without hesitation.",
]


def build_set(pos_templates, neg_templates, nouns):
    texts, labels = [], []
    for t in pos_templates:
        for n in nouns:
            texts.append(t.format(noun=n))
            labels.append(1)
    for t in neg_templates:
        for n in nouns:
            texts.append(t.format(noun=n))
            labels.append(0)
    return texts, np.array(labels)


id_texts, id_labels = build_set(ID_POS_TEMPLATES, ID_NEG_TEMPLATES, NOUNS)
ood_texts, ood_labels = build_set(OOD_POS_TEMPLATES, OOD_NEG_TEMPLATES, NOUNS)

# simple ID train/val split
rng = np.random.default_rng(0)
perm = rng.permutation(len(id_texts))
n_train = int(0.7 * len(id_texts))
train_idx, val_idx = perm[:n_train], perm[n_train:]

train_texts = [id_texts[i] for i in train_idx]
train_labels = id_labels[train_idx]
val_texts = [id_texts[i] for i in val_idx]
val_labels = id_labels[val_idx]

# ---------------------------------------------------------------------------
# 2. Model + SAE
# ---------------------------------------------------------------------------

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)

print("Loading pretrained SAE...")
# release/sae_id names per sae_lens; check sae_lens docs / neuronpedia if these
# have moved -- this is the standard gpt2-small residual stream SAE release.
sae, cfg_dict, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=f"blocks.{LAYER}.hook_resid_pre",
    device=DEVICE,
)

HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"


def get_activations(texts, batch_size=32):
    """Return last-token residual stream activations, shape (N, d_model)."""
    acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_NAME)
        resid = cache[HOOK_NAME]  # (batch, seq, d_model)
        # last real token per sequence (GPT-2 tokenizer left-pads? no -- right pad;
        # take the last non-pad position using attention mask via tokens != pad)
        last_pos = (tokens != model.tokenizer.pad_token_id).sum(dim=1) - 1
        for b in range(resid.shape[0]):
            acts.append(resid[b, last_pos[b]].cpu().numpy())
    return np.stack(acts)


print("Extracting activations...")
X_train = get_activations(train_texts)
X_val = get_activations(val_texts)
X_ood = get_activations(ood_texts)

# ---------------------------------------------------------------------------
# 3. SAE encode
# ---------------------------------------------------------------------------

def sae_encode(X):
    with torch.no_grad():
        feats = sae.encode(torch.tensor(X, device=DEVICE, dtype=torch.float32))
    return feats.cpu().numpy()


def sae_reconstruct(X):
    with torch.no_grad():
        feats = sae.encode(torch.tensor(X, device=DEVICE, dtype=torch.float32))
        recon = sae.decode(feats)
    return recon.cpu().numpy()


F_train = sae_encode(X_train)
F_val = sae_encode(X_val)
F_ood = sae_encode(X_ood)

Xhat_train = sae_reconstruct(X_train)
Xhat_val = sae_reconstruct(X_val)
Xhat_ood = sae_reconstruct(X_ood)

# ---------------------------------------------------------------------------
# 4. Probes
# ---------------------------------------------------------------------------

def fit_and_eval(X_tr, y_tr, X_va, y_va, X_ood_, y_ood_, name):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    tr_auc = roc_auc_score(y_tr, clf.decision_function(X_tr))
    va_auc = roc_auc_score(y_va, clf.decision_function(X_va))
    ood_auc = roc_auc_score(y_ood_, clf.decision_function(X_ood_))
    print(f"{name:30s} train={tr_auc:.3f}  val={va_auc:.3f}  OOD={ood_auc:.3f}")
    return clf


print("\n--- Dense linear probe (raw residual stream) ---")
fit_and_eval(X_train, train_labels, X_val, val_labels, X_ood, ood_labels,
             "Dense linear probe")

print("\n--- Linear probe on SAE reconstruction ---")
fit_and_eval(Xhat_train, train_labels, Xhat_val, val_labels, Xhat_ood, ood_labels,
             "Probe on SAE reconstruction")

print("\n--- Best single SAE latent ---")
# pick the latent with the largest mean-activation difference between classes
diff = F_train[train_labels == 1].mean(0) - F_train[train_labels == 0].mean(0)
best_latent = int(np.argmax(np.abs(diff)))
print(f"best latent index: {best_latent}")
fit_and_eval(F_train[:, [best_latent]], train_labels,
             F_val[:, [best_latent]], val_labels,
             F_ood[:, [best_latent]], ood_labels,
             "Single-latent SAE probe")

print("\n--- k-sparse SAE probe (top-k latents by mean diff) ---")
for k in [5, 20, 50]:
    top_k = np.argsort(-np.abs(diff))[:k]
    fit_and_eval(F_train[:, top_k], train_labels,
                 F_val[:, top_k], val_labels,
                 F_ood[:, top_k], ood_labels,
                 f"k={k} sparse SAE probe")