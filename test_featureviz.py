import torch

from activations import chunk_tokens
from featureviz import top_activating_examples

SEQ_LEN = 32


def _sequences():
    return chunk_tokens(list(range(50 * SEQ_LEN)), seq_len=SEQ_LEN)


def test_finds_the_strongest_firing_and_locates_its_token():
    z = torch.zeros(100, 4)
    z[67, 2] = 5.0
    z[13, 2] = 3.0

    examples = top_activating_examples(z, _sequences(), latent=2, top_n=2, context=3)

    assert [e.activation for e in examples] == [5.0, 3.0], "must be ranked strongest first"
    assert (examples[0].seq_index, examples[0].position) == (2, 3), "67 = seq 2, pos 3"
    assert (examples[1].seq_index, examples[1].position) == (0, 13)


def test_context_window_surrounds_the_firing_token():
    z = torch.zeros(100, 4)
    z[67, 2] = 5.0

    example = top_activating_examples(z, _sequences(), latent=2, top_n=1, context=3)[0]

    # activation 67 is token id 67; a window of 3 either side is 64..70
    assert example.tokens == list(range(64, 71))
    assert example.tokens[example.peak_offset] == 67


def test_context_window_clips_at_the_sequence_start():
    """Token 1 has no room for 3 tokens of left context; the window must not
    bleed into the previous sequence, which is unrelated text."""
    z = torch.zeros(100, 4)
    z[1, 2] = 5.0

    example = top_activating_examples(z, _sequences(), latent=2, top_n=1, context=3)[0]

    assert example.tokens == [0, 1, 2, 3, 4]
    assert example.tokens[example.peak_offset] == 1


def test_ignores_latents_that_never_fire():
    z = torch.zeros(100, 4)

    examples = top_activating_examples(z, _sequences(), latent=3, top_n=5, context=3)

    assert examples == [], "a dead latent has no examples, not a list of zeros"


from featureviz import render_examples


def _decode(token_id):
    return {64: "the", 65: " Golden", 66: " Gate", 67: " Bridge", 68: " is"}.get(
        token_id, f"<{token_id}>"
    )


def test_render_highlights_only_the_firing_token():
    z = torch.zeros(100, 4)
    z[67, 2] = 5.0
    examples = top_activating_examples(z, _sequences(), latent=2, top_n=1, context=3)

    html = render_examples(examples, decode_token=_decode, latent=2)

    assert "<mark" in html
    assert "Bridge</mark>" in html, "the peak token must be the highlighted one"
    assert html.count("<mark") == 1, "exactly one token highlighted per example"
    assert "5.0" in html, "the activation value should be shown"
