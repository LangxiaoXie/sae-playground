"""Answering "what does this latent mean?" by showing the text that lights it up.

An SAE latent is just a column index. It becomes interpretable only when you
look at which tokens make it fire hardest -- so this module maps latent
activations back to the text positions that produced them.
"""

from dataclasses import dataclass

import torch


@dataclass
class Example:
    """One firing of a latent, with enough context to read it.

    tokens/peak_offset are token ids rather than text so that the index maths
    stays testable without dragging a tokenizer into it; rendering happens
    separately.
    """

    seq_index: int
    position: int
    activation: float
    tokens: list[int]
    peak_offset: int


def top_activating_examples(
    z: torch.Tensor,
    sequences: torch.Tensor,
    latent: int,
    top_n: int = 10,
    context: int = 8,
) -> list[Example]:
    """Find where `latent` fired hardest, and grab the surrounding tokens.

    Firings of exactly zero are dropped rather than ranked, so a dead latent
    returns nothing at all instead of an arbitrary list of zero-valued hits.
    """
    seq_len = sequences.shape[1]
    column = z[:, latent]

    nonzero = (column > 0).nonzero(as_tuple=True)[0]
    if nonzero.numel() == 0:
        return []

    order = column[nonzero].argsort(descending=True)[:top_n]
    hits = nonzero[order]

    examples = []
    for flat_index in hits.tolist():
        seq_index, position = divmod(flat_index, seq_len)
        # Clip to the sequence: a window that ran past either end would splice in
        # tokens from unrelated text and make the latent look like it fires on things
        # it never saw.
        start = max(0, position - context)
        stop = min(seq_len, position + context + 1)
        window = sequences[seq_index, start:stop].tolist()

        examples.append(
            Example(
                seq_index=seq_index,
                position=position,
                activation=float(column[flat_index]),
                tokens=window,
                peak_offset=position - start,
            )
        )
    return examples


def render_examples(examples: list[Example], decode_token, latent: int | None = None) -> str:
    """Render firings as HTML with the responsible token highlighted.

    `decode_token` maps one token id to its string, injected rather than taking
    a whole model so the layout logic stays independent of any tokenizer.
    """
    from html import escape

    if not examples:
        return f"<p><em>latent {latent} never fires (dead)</em></p>"

    blocks = []
    if latent is not None:
        blocks.append(f"<h4 style='font-family:sans-serif;margin:.6em 0 .2em'>latent {latent}</h4>")

    for e in examples:
        rendered = []
        for offset, token in enumerate(e.tokens):
            text = escape(decode_token(token))
            if offset == e.peak_offset:
                rendered.append(
                    f"<mark style='background:#ffd54f;font-weight:600;border-radius:2px'>{text}</mark>"
                )
            else:
                rendered.append(text)
        blocks.append(
            "<div style='font-family:ui-monospace,monospace;font-size:12px;"
            "padding:3px 6px;border-left:3px solid #ffd54f;margin:2px 0'>"
            f"<span style='color:#888'>{e.activation:.2f}</span>&nbsp; "
            f"{''.join(rendered)}</div>"
        )
    return "\n".join(blocks)
