"""TCP: the invariants that make it Boltz-Perturb's intervention and not ours.

Every assertion here is a property stated in their paper or visible in their code;
if the port drifts, the experiment stops being a replication and these fail first.
"""
from __future__ import annotations



import pytest

torch = pytest.importorskip("torch")

from cofolding_tts import tcp as TCP


def block(seed=0, n=40, lig_from=30, c=16):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(1, n, c, generator=g) * 3.0 + 1.5
    lig = torch.zeros(n, dtype=torch.bool)
    lig[lig_from:] = True
    eps = torch.randn(1, int(lig.sum()), c, generator=g)
    return s, lig, eps


# ---------------------------------------------------------------- the ramp

def test_window_maps_to_the_steps_the_paper_names():
    """C11 is t in [0.2, 0.6]; with 200 steps that has to be steps 80-160."""
    lam = [TCP.lam_at(i, 200, 0.9, 0.6, 0.2) for i in range(200)]
    fired = [i for i, v in enumerate(lam) if v > 0]
    assert fired[0] == 80 and fired[-1] == 159
    assert lam[79] == 0.0 and lam[160] == 0.0


def test_lambda_is_largest_at_the_early_edge():
    """Their ramp fades as the pose commits. Getting this backwards would perturb
    hardest exactly where our commitment window sits, which is a different
    experiment."""
    lam = [TCP.lam_at(i, 200, 0.9, 0.6, 0.2) for i in range(80, 160)]
    assert lam[0] == pytest.approx(0.9, abs=1e-6)
    assert lam == sorted(lam, reverse=True)
    assert lam[-1] < 0.02


def test_zero_lambda_is_a_no_op():
    s, lig, eps = block()
    assert torch.equal(TCP.tcp(s, lig, eps, 0.0, 0.7), s)


# ------------------------------------------------- what TCP does and does not touch

def test_protein_rows_are_untouched():
    s, lig, eps = block()
    out = TCP.tcp(s, lig, eps, 0.9, 0.7)
    assert torch.equal(out[:, ~lig, :], s[:, ~lig, :])
    assert not torch.allclose(out[:, lig, :], s[:, lig, :])


def test_norm_projection_preserves_every_ligand_row_norm():
    """Step 5 of their function. This is why TCP is a ROTATION: no row changes
    length, so the only thing the perturbation can do is change direction."""
    s, lig, eps = block()
    out = TCP.tcp(s, lig, eps, 0.9, 0.7, norm_proj=True)
    before = s[:, lig, :].norm(dim=-1)
    after = out[:, lig, :].norm(dim=-1)
    assert torch.allclose(before, after, rtol=1e-4)


def test_without_norm_projection_the_rows_change_length():
    s, lig, eps = block()
    out = TCP.tcp(s, lig, eps, 0.9, 0.7, norm_proj=False)
    assert not torch.allclose(s[:, lig, :].norm(dim=-1),
                              out[:, lig, :].norm(dim=-1), rtol=1e-3)


def test_rescale_restores_the_block_moments_before_interpolation():
    """Step 3 matches mean and std of the ligand block to the unperturbed one."""
    s, lig, eps = block()
    out = TCP.tcp(s, lig, eps, 0.9, 1.0, rescale=True, norm_proj=False)
    a, b = s[:, lig, :], out[:, lig, :]
    assert b.mean() == pytest.approx(float(a.mean()), abs=1e-3)
    assert b.std() == pytest.approx(float(a.std()), abs=1e-3)


def test_psi_interpolates_toward_the_original():
    """psi = 0 must return s exactly; larger psi must move further."""
    s, lig, eps = block()
    assert torch.allclose(TCP.tcp(s, lig, eps, 0.9, 0.0, norm_proj=False),
                          s, atol=1e-5)
    d = [TCP.geometry(s[:, lig, :],
                      TCP.tcp(s, lig, eps, 0.9, p)[:, lig, :])["rotation_deg"]
         for p in (0.2, 0.7, 1.0)]
    assert d == sorted(d)


def test_the_perturbation_is_a_modest_rotation_at_their_settings():
    """The claim the experiment is built on: TCP at s=0.9, psi=0.7 moves the ligand
    rows by tens of degrees, not by the ~88 degrees a full gaussian replacement
    does. If this ever reads near 90, the two arms are no longer distinguishable
    and the comparison in experiment.yaml is void."""
    s, lig, eps = block()
    g = TCP.geometry(s[:, lig, :], TCP.tcp(s, lig, eps, 0.9, 0.7)[:, lig, :])
    assert 5.0 < g["rotation_deg"] < 60.0


def test_noise_scale_is_taken_over_the_whole_tensor_not_the_ligand_block():
    """Their line 314 uses s.std() over all tokens. A ligand-only std would be a
    different intervention whenever the two blocks have different spread."""
    n, c = 40, 16
    s = torch.zeros(1, n, c)
    g = torch.Generator().manual_seed(1)
    s[:, :30, :] = torch.randn(1, 30, c, generator=g) * 10.0   # loud protein
    s[:, 30:, :] = torch.randn(1, 10, c, generator=g) * 0.1    # quiet ligand
    lig = torch.zeros(n, dtype=torch.bool); lig[30:] = True
    eps = torch.randn(1, 10, c, generator=g)
    rot = TCP.geometry(s[:, lig, :],
                       TCP.tcp(s, lig, eps, 0.9, 1.0, rescale=False,
                               norm_proj=False)[:, lig, :])["rotation_deg"]
    # scaled by the loud global std, the ligand rows are swamped -> a large angle
    assert rot > 60.0
