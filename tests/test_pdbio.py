"""PDB records are fixed-column. These tests pin every field to its column.

A one-column shift here silently moves the coordinates into the wrong fields.
Whitespace-splitting parsers still read it, so the error survives inspection;
column-slicing parsers read garbage. That happened once and is why this exists.
"""

from __future__ import annotations

import pytest

from cofolding_tts.pdbio import ligand_block, record


@pytest.fixture
def line():
    return record("ATOM", 1, "CA", "LEU", "A", 191, (12.345, -6.789, 30.1))


def test_line_is_the_standard_width(line):
    assert len(line) == 78


def test_every_field_is_in_its_column(line):
    assert line[0:6] == "ATOM  "
    assert line[6:11] == "    1"
    assert line[12:16] == " CA "
    assert line[17:20] == "LEU"
    assert line[21] == "A"
    assert line[22:26] == " 191"
    assert line[30:38] == "  12.345"
    assert line[38:46] == "  -6.789"
    assert line[46:54] == "  30.100"
    assert line[76:78] == " C"


def test_two_character_elements_start_one_column_earlier():
    line = record("HETATM", 2, "BR", "LIG", "L", 3, (1.0, 2.0, 3.0))
    assert line[12:16] == "BR  "
    assert line[76:78] == "BR"
    assert line[30:38] == "   1.000"


def test_element_is_inferred_but_can_be_given():
    assert record("ATOM", 1, "CB", "ALA", "A", 1, (0, 0, 0))[76:78] == " C"
    assert record("ATOM", 1, "CB", "ALA", "A", 1, (0, 0, 0), element="N")[76:78] == " N"


def test_occupancy_and_bfactor_columns():
    line = record("ATOM", 1, "CA", "LEU", "A", 1, (0, 0, 0), occupancy=0.5, bfactor=12.25)
    assert line[54:60] == "  0.50"
    assert line[60:66] == " 12.25"


def test_ligand_block_numbers_atoms_from_one():
    block = ligand_block(["N", "CA", "C"], [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    lines = block.splitlines()
    assert len(lines) == 3
    assert [l[6:11] for l in lines] == ["    1", "    2", "    3"]
    assert all(l.startswith("HETATM") for l in lines)


def test_ligand_block_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ligand_block(["N", "CA"], [(0, 0, 0)])
