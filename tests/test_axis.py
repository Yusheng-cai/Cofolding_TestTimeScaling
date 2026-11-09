"""Tests for candidate stereogenic-axis perception.

The cases that matter are the NEGATIVES. A detector that fires on every biaryl
is useless for screening a benchmark, and the thing that separates a real
stereogenic axis from an ordinary rotatable biaryl is local symmetry, not
bulk: biphenyl has four free ortho positions and no stereogenic axis, while
2-methylbiphenyl has a substituent and still has none, because the far ring maps
onto itself under the 180-degree rotation.
"""
from rdkit import Chem

from cofolding_tts.axis import find_axis_candidates, has_axis

# the 8SGE ligand ZWS, exactly as the exp096 manifest supplies it
ZWS = "CC[C@@H](C)c1cc2c(n(-c3ccc(=O)n(CC(=O)O)c3)c1=O)CN(C(=O)OCCOC)CC2"


def test_detects_the_8sge_axis():
    """The axis this whole line of work is about must be found."""
    found = find_axis_candidates(Chem.MolFromSmiles(ZWS))
    assert len(found) == 1
    assert found[0].kind in ("N-C", "C-N")


def test_detects_classic_atropisomers():
    for smiles in (
        "Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12",                 # BINOL
        "Cc1cccc([N+](=O)[O-])c1-c1c(C)cccc1[N+](=O)[O-]",   # 2,2'-NO2 6,6'-Me biphenyl
        # telmisartan, an atropisomeric marketed drug
        "CCCc1nc2c(C)cc(-c3nc4ccccc4n3C)cc2n1Cc1ccc(-c2ccccc2C(=O)O)cc1",
    ):
        assert has_axis(smiles), smiles


def test_rejects_symmetric_biaryls():
    """Rotating these by 180 degrees maps the molecule onto itself."""
    assert not has_axis("c1ccccc1-c1ccccc1")          # biphenyl
    assert not has_axis("Cc1ccccc1-c1ccccc1")         # 2-methylbiphenyl: far ring symmetric
    assert not has_axis("c1ccc(-c2ccncc2)cc1")        # phenylpyridine, symmetric ring


def test_rejects_non_axis_bonds():
    assert not has_axis("CCO")                        # no rings
    assert not has_axis("c1ccccc1C")                   # ring-to-chain, not ring-to-ring
    assert not has_axis("C1CCCCC1-C1CCCCC1")           # saturated, not aromatic
    assert not has_axis("c1ccc2ccccc2c1")              # fused: the bond is IN a ring


def test_blocked_ortho_is_counted():
    """The steric proxy should rank a tetra-ortho case above a mono-ortho one."""
    tetra = find_axis_candidates(Chem.MolFromSmiles(
        "Cc1cccc([N+](=O)[O-])c1-c1c(C)cccc1[N+](=O)[O-]"))[0]
    binol = find_axis_candidates(Chem.MolFromSmiles(
        "Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12"))[0]
    assert tetra.blocked_ortho == 4
    assert 0 <= binol.blocked_ortho <= 4
    assert binol.fused_either_side is True


def test_handles_bad_input():
    assert find_axis_candidates(None) == []
    assert not has_axis("not a smiles")


def test_result_is_serialisable():
    c = find_axis_candidates(Chem.MolFromSmiles(ZWS))[0]
    d = c.as_dict()
    assert d["bond_idx"] == c.bond_idx
    assert set(d) == {"bond_idx", "begin_idx", "end_idx", "kind", "blocked_ortho",
                      "begin_ring_size", "end_ring_size", "fused_either_side"}
