"""Plot rendering for a terminal, using coloured block characters.

Notebook 1 uses plotly, which needs a browser. These render straight into the
terminal instead, so the whole walkthrough works in a split with no extra deps.
"""

import shutil

RESET = "\033[0m"


def _bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


def _diverging(v: float, gamma: float = 1.0) -> str:
    """-1 blue, 0 near-black, +1 warm yellow. v is clamped to [-1, 1].

    gamma > 1 darkens mid-range values. Random unit vectors in a low-dimensional
    space sit around |cos| 0.4, which on a linear scale glows nearly as brightly
    as a real 0.95 match and hides the signal.
    """
    v = max(-1.0, min(1.0, v))
    if gamma != 1.0:
        v = (abs(v) ** gamma) * (1 if v >= 0 else -1)
    if v >= 0:
        r, g, b = int(20 + 235 * v), int(20 + 195 * v), int(30 + 25 * v)
    else:
        r, g, b = int(20 - 5 * v), int(20 - 60 * v), int(30 - 195 * v)
    return _bg(r, g, b)


def heatmap(matrix, row_labels=None, col_label="", title="", highlight=None,
            gamma: float = 1.0) -> None:
    """Draw a 2-D array as coloured blocks, one cell per two character columns."""
    rows = len(matrix)
    cols = len(matrix[0])
    if title:
        print(f"\n\033[1m{title}\033[0m")

    width = shutil.get_terminal_size((100, 24)).columns
    step = max(1, -(-cols * 2 // max(20, width - 12)))

    for i in range(rows):
        label = (row_labels[i] if row_labels else str(i)).rjust(8)
        cells = "".join(_diverging(float(matrix[i][j]), gamma) + "  " for j in range(0, cols, step))
        marker = ""
        if highlight is not None:
            marker = f" {RESET}\033[2m← latent {highlight[i]}\033[0m"
        print(f"{label} {cells}{RESET}{marker}")

    print(f"{'':8} \033[2m{col_label}{'' if step == 1 else f'  (every {step}nd column)'}\033[0m")
    scale = "".join(_diverging(-1 + 2 * i / 24, gamma) + " " for i in range(25))
    note = "" if gamma == 1.0 else f"  (contrast \u03b3={gamma:g})"
    print(f"{'':8} {scale}{RESET} \033[2m-1 .. 0 .. +1{note}\033[0m")


def barplot(labels, values, title="", width=44, fmt="{:.2f}", highlight_max=True) -> None:
    """Horizontal bars, scaled to the widest value."""
    if title:
        print(f"\n\033[1m{title}\033[0m")
    top = max(values) or 1.0
    best = max(range(len(values)), key=lambda i: values[i]) if highlight_max else -1
    for label, value in zip(labels, values):
        filled = int(width * value / top)
        colour = "\033[38;5;220m" if labels.index(label) == best else "\033[38;5;110m"
        print(f"  {str(label):>8} {colour}{'█' * filled}{'░' * (width - filled)}{RESET} {fmt.format(value)}")


def scatter(xs, ys, labels, title="", xlabel="", ylabel="", height=13, width=52) -> None:
    """A tiny scatter plot on a character grid."""
    if title:
        print(f"\n\033[1m{title}\033[0m")
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    span_x = (hi_x - lo_x) or 1.0
    span_y = (hi_y - lo_y) or 1.0

    grid = [[" "] * width for _ in range(height)]
    for x, y, lab in zip(xs, ys, labels):
        col = int((x - lo_x) / span_x * (width - 1))
        row = height - 1 - int((y - lo_y) / span_y * (height - 1))
        grid[row][col] = "\033[38;5;220m●\033[0m"
        for k, ch in enumerate(str(lab)):
            if col + 2 + k < width:
                grid[row][col + 2 + k] = f"\033[2m{ch}\033[0m"

    for r, line in enumerate(grid):
        axis = f"{hi_y:>7.1f}" if r == 0 else (f"{lo_y:>7.1f}" if r == height - 1 else " " * 7)
        print(f"{axis} │{''.join(line)}")
    print(f"{'':7} └{'─' * width}")
    print(f"{'':8}{lo_x:<.1f}{ylabel:^{width - 10}}{hi_x:>.1f}")
    print(f"{'':8}\033[2m{xlabel}\033[0m")
