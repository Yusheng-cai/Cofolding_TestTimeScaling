"""The LaTeX-to-plain fallback in figstyle.tex().

Figures must still render on a clone with no LaTeX installed, so every label is
written in LaTeX markup and rewritten when text.usetex is off. That rewrite is a
small parser, which is why it is pinned here. These tests import no matplotlib,
which is also the point of figstyle importing it lazily.
"""

from cofolding_tts import figstyle


def plain(s: str) -> str:
    return figstyle.tex(s, usetex=False)


def test_latex_mode_passes_markup_through():
    assert figstyle.tex(r"\textbf{(a)} Boltz-2", usetex=True) == r"\textbf{(a)} Boltz-2"


def test_angstrom_and_percent_become_literal():
    assert plain(r"RMSD $<$ 2\,\AA") == "RMSD $<$ 2\u2009\u00c5"
    assert plain(r"25\%") == "25%"


def test_textbf_is_unwrapped_not_dropped():
    assert plain(r"\textbf{(a)} Boltz-2") == "(a) Boltz-2"


def test_nested_braces_inside_textbf_survive():
    # \textbf{(a)} often arrives from an f-string with braces of its own; a naive
    # index-of-"}" would cut at the wrong one as soon as anything nests.
    assert plain(r"\textbf{x{y}z} tail") == "x{y}z tail"


def test_several_textbf_in_one_string():
    assert plain(r"\textbf{a} and \textbf{b}") == "a and b"


def test_unmatched_textbf_does_not_hang_or_raise():
    assert plain(r"\textbf{a") == "a"


def test_default_is_plain_until_use_is_called():
    assert figstyle.tex(r"\textbf{a}") == "a"
