"""Feeding real GPT-2 activations to the SAE.

Stage 1 hands the SAE synthetic vectors whose causes we planted. This module
swaps in the real thing: the residual stream of GPT-2, read at one layer, over
real text. Nothing about the SAE changes -- only where its input comes from.
"""

import torch


def chunk_tokens(tokens: list[int], seq_len: int) -> torch.Tensor:
    """Cut a flat token stream into equal-length sequences, dropping the remainder.

    The tail is discarded rather than padded: padding would feed the SAE
    activations for positions that are not real text, and those show up later
    as spurious "features" that only fire on padding.
    """
    n_full = len(tokens) // seq_len
    usable = torch.tensor(tokens[: n_full * seq_len], dtype=torch.long)
    return usable.view(n_full, seq_len)


@torch.no_grad()
def collect_activations(
    model,
    sequences: torch.Tensor,
    hook_name: str,
    n_activations: int,
    batch_size: int = 8,
    dtype: torch.dtype = torch.float16,
    progress: bool = False,
) -> torch.Tensor:
    """Run text through the model and keep the activations at one hook point.

    Every token position yields one vector, so a batch of 8 sequences of length
    128 produces 1024 activations. Returns (n_activations, d_model) in fp16,
    which halves the memory for no measurable cost to SAE quality.
    """
    collected: list[torch.Tensor] = []
    total = 0

    iterator = range(0, len(sequences), batch_size)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="collecting activations")

    for start in iterator:
        batch = sequences[start : start + batch_size].to(model.cfg.device)
        _, cache = model.run_with_cache(batch, names_filter=hook_name)
        acts = cache[hook_name].flatten(0, 1)  # (batch * seq, d_model)

        collected.append(acts[: n_activations - total].to(dtype).cpu())
        total += collected[-1].shape[0]
        if total >= n_activations:
            break

    if total < n_activations:
        raise ValueError(
            f"corpus exhausted at {total} activations, needed {n_activations}; "
            f"pass more sequences"
        )
    return torch.cat(collected)


def save_activations(path, acts: torch.Tensor, hook_name: str, seq_len: int) -> None:
    """Cache activations to disk alongside the settings that produced them.

    The metadata is not bookkeeping for its own sake: training an SAE on a
    layer-6 cache and then browsing its features against layer-8 text produces
    plausible-looking nonsense, and this is what makes that mistake loud.
    """
    torch.save(
        {"acts": acts, "meta": {"hook_name": hook_name, "seq_len": seq_len}}, path
    )


def load_activations(path) -> tuple[torch.Tensor, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["acts"], blob["meta"]


def load_corpus_sequences(
    model,
    n_sequences: int,
    seq_len: int = 128,
    dataset_name: str = "NeelNanda/pile-10k",
    max_tokens_per_doc: int = 2048,
    seed: int = 0,
) -> torch.Tensor:
    """Tokenise real text into a (n_sequences, seq_len) batch.

    Documents are concatenated into one stream before chunking, so a sequence
    may span a document boundary. That is standard for SAE training -- the SAE
    sees activations, not documents -- and keeps every sequence full length.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))

    needed = n_sequences * seq_len
    tokens: list[int] = []
    for index in order.tolist():
        ids = model.to_tokens(dataset[index]["text"], prepend_bos=False)[0]
        tokens.extend(ids[:max_tokens_per_doc].tolist())
        if len(tokens) >= needed:
            break

    if len(tokens) < needed:
        raise ValueError(f"corpus gave {len(tokens)} tokens, needed {needed}")
    return chunk_tokens(tokens[:needed], seq_len)
