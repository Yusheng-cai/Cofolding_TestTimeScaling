"""Column-correct PDB record writer.

The PDB format is fixed-column, not whitespace-delimited. Writing `name` and
`resName` adjacent without the altLoc column shifts every later field left by
one, which silently moves the coordinates into the wrong columns. Parsers that
slice by column then read garbage, and parsers that split on whitespace do not,
so the bug survives casual inspection. It cost a debugging session once; this
module exists so it cannot happen again.

Columns (1-indexed, inclusive):
    1-6   record name       7-11  serial        13-16 atom name
    17    altLoc            18-20 resName       22    chainID
    23-26 resSeq            27    iCode         31-38 x
    39-46 y                 47-54 z             55-60 occupancy
    61-66 bFactor           77-78 element
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["record", "ligand_block"]


def record(
    kind: str,
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    xyz: Sequence[float],
    *,
    occupancy: float = 1.0,
    bfactor: float = 0.0,
    element: str | None = None,
) -> str:
    """One ATOM or HETATM line, 78 characters, every field in its column."""

    el = (element or (name[:2] if name[:2].upper() == "BR" else name[:1])).upper()
    nm = name[:4]
    # Two-character element symbols start in column 13; one-character names are
    # conventionally indented to column 14.
    name_field = f"{nm:<4s}" if len(el) == 2 else f" {nm:<3s}"
    return (
        f"{kind:<6s}{serial:5d} {name_field}"
        f" {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {el:>2s}"
    )


def ligand_block(
    names: Sequence[str],
    coords,
    *,
    resname: str = "LIG",
    chain: str = "L",
    resseq: int = 1,
    bfactor: float = 0.0,
) -> str:
    """HETATM lines for one ligand copy, newline-joined, no END."""

    if len(names) != len(coords):
        raise ValueError(f"{len(names)} atom names for {len(coords)} coordinates")
    return "\n".join(
        record("HETATM", i, n, resname, chain, resseq, p, bfactor=bfactor)
        for i, (n, p) in enumerate(zip(names, coords), start=1)
    )
