"""[inspect] Shared manuscript figure style.

Computer Modern text, boxed axes with inward ticks on all four sides, bold
parenthesised panel labels, large tick and label text, large filled markers.
This matches the style the project's manuscript figures are drawn in, so a plot
taken from this repository drops into a talk or a paper without restyling.

LaTeX is used for text when a ``latex`` binary is on PATH and falls back to
matplotlib's own Computer Modern mathtext when it is not, so the inspect tier
still runs on a bare clone with nothing but numpy and matplotlib installed.
Call :func:`tex` to write a string that is correct either way.

    from cofolding_tts import figstyle
    figstyle.use()
    fig, ax = plt.subplots(figsize=(7, 5))
    figstyle.box(ax)
    figstyle.panel(fig, ax, "a")
"""

from __future__ import annotations

import shutil

__all__ = ["use", "box", "panel", "tex", "usetex", "INK", "MUTED", "SURFACE",
           "BLUE", "ORANGE", "GREEN", "VERM", "PURPLE", "PINK"]

INK, MUTED, SURFACE = "#1a1a1a", "#6e6e6e", "#ffffff"
# Okabe-Ito, assigned in a fixed order and never cycled, plus the manuscript's
# accent pink reserved for a single highlighted point.
BLUE, ORANGE, GREEN, VERM, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
PINK = "#C2187A"

# matplotlib is imported lazily, inside the functions that draw, so that `tex()`
# and the palette can be used (and tested) without it.
_USETEX: bool = False


def usetex() -> bool:
    """True when :func:`use` installed a style that renders text through LaTeX."""
    return _USETEX


def tex(s: str, usetex: bool | None = None) -> str:
    r"""Make a label correct under both LaTeX and mathtext.

    Write labels with LaTeX markup (``\AA``, ``\textbf{..}``, ``\%``); when no
    LaTeX binary is available this rewrites them into plain text so the figure
    still renders rather than raising. ``usetex`` overrides what :func:`use`
    installed, which is what the tests pin.
    """
    if _USETEX if usetex is None else usetex:
        return s
    for src, dst in ((r"\AA", "Å"), (r"\%", "%"), (r"\,", " "),
                     (r"\textendash", "–"), (r"\times", "×")):
        s = s.replace(src, dst)
    while r"\textbf{" in s:                       # drop one level of \textbf{..}
        i = s.index(r"\textbf{")
        depth, j = 0, i + 8
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        s = s[:i] + s[i + 8:j] + s[j + 1:]
    return s


def use(base: float = 15.0, *, force_usetex: bool | None = None) -> bool:
    """Install the style. ``base`` is the body font size in points.

    Tick labels sit at ``base``, axis labels a little above it, and panel labels
    well above, which is the ratio the manuscript figures use. Returns whether
    LaTeX is in play.
    """
    global _USETEX
    import matplotlib
    import matplotlib.pyplot as plt

    want = shutil.which("latex") is not None if force_usetex is None else force_usetex
    matplotlib.rcdefaults()
    plt.rcParams.update({
        "text.usetex": want,
        "font.family": "serif",
        # Under LaTeX the CM fonts are complete. Without it, matplotlib's own
        # cmr10 has no Angstrom sign and would silently render a tofu box, so
        # the fallback leads with a serif face that covers the glyphs we use.
        "font.serif": (["Computer Modern Roman", "cmr10", "DejaVu Serif"] if want
                       else ["DejaVu Serif", "cmr10"]),
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": base,
        "axes.labelsize": base * 1.2,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base * 0.95,
        "axes.linewidth": 1.4,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 2.2,
        "lines.markersize": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "standard",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if want:
        # cmbright is not assumed; only the base CM fonts and \AA are needed.
        plt.rcParams["text.latex.preamble"] = ""
    _USETEX = want
    return want


def box(ax, *, length: float = 7.0, width: float = 1.3, pad: float = 7.0,
        minor: bool = True) -> None:
    """Boxed axes with inward ticks on all four sides."""
    import matplotlib.pyplot as plt

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(plt.rcParams["axes.linewidth"])
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   length=length, width=width, pad=pad)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       length=length * 0.5, width=width * 0.8)


def panel(fig, ax, letter: str, *, dx: float = -0.085, dy: float = 0.022,
          size: float | None = None, text: str = "") -> None:
    """Bold parenthesised panel label, placed outside the axes' top-left corner.

    ``dx``/``dy`` are offsets in figure fractions from that corner; ``text`` is
    an optional short caption set beside the label in the normal weight.
    """
    import matplotlib.pyplot as plt

    bb = ax.get_position()
    size = size or plt.rcParams["font.size"] * 1.35
    label = rf"\textbf{{({letter})}}"
    if text:
        label += f" {text}"
    fig.text(bb.x0 + dx, bb.y1 + dy, tex(label), fontsize=size, color=INK,
             ha="left", va="bottom")
