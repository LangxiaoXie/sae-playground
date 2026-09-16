"""One-time build of the GPT-2 activation cache used by 02_gpt2.ipynb.

Pure composition over the tested functions in activations.py -- run it once,
then every SAE you train reads from the cache instead of re-running GPT-2.

    python build_cache.py --n 500000

Delete acts_layer6.pt when you are done; nothing else on disk grows.
"""

import argparse
import time

import torch

from activations import collect_activations, load_corpus_sequences, save_activations


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500_000, help="activations to cache")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="acts_layer6.pt")
    args = parser.parse_args()

    hook_name = f"blocks.{args.layer}.hook_resid_pre"
    device = pick_device()
    print(f"device={device} hook={hook_name} target={args.n:,} activations", flush=True)

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained("gpt2", device=device)

    n_sequences = -(-args.n // args.seq_len) + args.batch_size
    print(f"tokenising {n_sequences:,} sequences of {args.seq_len} tokens...", flush=True)
    sequences = load_corpus_sequences(model, n_sequences, seq_len=args.seq_len)

    start = time.time()
    acts = collect_activations(
        model, sequences, hook_name, args.n, batch_size=args.batch_size, progress=True
    )
    elapsed = time.time() - start

    save_activations(args.out, acts, hook_name=hook_name, seq_len=args.seq_len)
    torch.save(sequences[: -(-args.n // args.seq_len)], args.out.replace(".pt", "_seqs.pt"))

    size_mb = acts.numel() * acts.element_size() / 1e6
    print(f"\n{tuple(acts.shape)} {acts.dtype} -> {args.out} ({size_mb:.0f} MB) "
          f"in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
