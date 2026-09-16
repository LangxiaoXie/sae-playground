import torch

from activations import chunk_tokens


def test_chunks_a_token_stream_into_fixed_length_sequences():
    tokens = list(range(100))

    chunks = chunk_tokens(tokens, seq_len=32)

    assert chunks.shape == (3, 32), "100 tokens at seq_len 32 gives 3 full sequences"
    assert chunks[0].tolist() == list(range(32))
    assert chunks[2].tolist() == list(range(64, 96))


def test_discards_the_ragged_tail_rather_than_padding_it():
    """Padding would inject activations for tokens that are not real text."""
    chunks = chunk_tokens(list(range(70)), seq_len=32)
    assert chunks.shape == (2, 32)


import pytest

HOOK = "blocks.6.hook_resid_pre"


@pytest.mark.slow
def test_collect_returns_one_activation_vector_per_token(gpt2):
    from activations import collect_activations

    sequences = chunk_tokens(list(range(50 * 32)), seq_len=32)

    acts = collect_activations(gpt2, sequences, HOOK, n_activations=100, batch_size=8)

    assert acts.shape == (100, 768)


@pytest.mark.slow
def test_collected_activations_vary_across_positions(gpt2):
    """Guards against silently collecting the same vector 100 times."""
    from activations import collect_activations

    sequences = chunk_tokens(list(range(50 * 32)), seq_len=32)

    acts = collect_activations(gpt2, sequences, HOOK, n_activations=100, batch_size=8)

    assert acts.float().std(dim=0).mean() > 0.1


def test_activation_cache_round_trips_with_its_metadata(tmp_path):
    from activations import save_activations, load_activations

    acts = torch.randn(100, 8, dtype=torch.float16)
    path = tmp_path / "acts.pt"

    save_activations(path, acts, hook_name=HOOK, seq_len=32)
    loaded, meta = load_activations(path)

    assert torch.equal(loaded, acts)
    assert meta["hook_name"] == HOOK
    assert meta["seq_len"] == 32


@pytest.mark.slow
def test_activation_n_maps_back_to_sequence_and_position(gpt2):
    """The feature browser needs this: given a latent that fired at activation
    index i, show the text around it. That only works if index arithmetic is
    exact, so pin it down."""
    from activations import collect_activations

    seq_len = 32
    sequences = chunk_tokens(list(range(50 * seq_len)), seq_len=seq_len)
    acts = collect_activations(gpt2, sequences, HOOK, n_activations=100, batch_size=8)

    i = 67
    _, cache = gpt2.run_with_cache(sequences[i // seq_len : i // seq_len + 1],
                                   names_filter=HOOK)
    expected = cache[HOOK][0, i % seq_len].half()

    assert torch.allclose(acts[i], expected, atol=1e-2)


@pytest.mark.slow
def test_load_corpus_returns_real_token_sequences(gpt2):
    from activations import load_corpus_sequences

    seqs = load_corpus_sequences(gpt2, n_sequences=20, seq_len=64)

    assert seqs.shape == (20, 64)
    assert seqs.dtype == torch.long
    assert seqs.max() < gpt2.cfg.d_vocab, "token ids must be in the model's vocabulary"
    assert seqs.float().std() > 0, "not all the same token"
