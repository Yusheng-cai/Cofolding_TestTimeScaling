"""Focused tests for the exp097 conditioning interventions.

Two things must hold. First, each intervention has to do what its name says:
`additive` buries the signal under lambda-scaled noise but leaves it present,
while `mean` and `gauss` destroy token-to-token variation. Second, adding the
destructive interventions must not disturb the additive path, because 151 runs
from phases A, A2 and A3 are on disk and have to stay bit-reproducible.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[1] / "src/cofolding_tts/intervene.py"
)
SPEC = importlib.util.spec_from_file_location("cofolding_tts_intervene", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

WHOLE = (slice(None), slice(None), slice(None))


def apply(intervention, tensor, lam=0.0, key="s_lig", base_seed=7,
          n_channels=0, channel_order="top"):
    """Run one `bump` in isolation, returning the new tensor and its log entry."""
    log: list[dict] = []
    installed = {}

    class FakeDiffusion:
        @staticmethod
        def sample(*_args, **_kwargs):
            return None

    # install() reaches into boltz; exercise bump() directly instead by rebuilding
    # the same closure through a stub module
    import sys
    import types

    stub = types.ModuleType("boltz.model.modules.diffusionv2")
    stub.AtomDiffusion = FakeDiffusion
    for name in ("boltz", "boltz.model", "boltz.model.modules"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["boltz.model.modules.diffusionv2"] = stub

    MODULE.install(("s_lig",), lam, base_seed, False, log, (), intervention,
                   None, False, "sample", n_channels, channel_order)
    installed["patched"] = FakeDiffusion.sample

    # bump is a closure inside install(); reach it through the patched function's
    # own closure so the test exercises the shipped code rather than a copy
    cells = installed["patched"].__closure__
    bump = next(c.cell_contents for c in cells
                if getattr(c.cell_contents, "__name__", "") == "bump")
    out = tensor.clone()
    bump(out, WHOLE, key, "test")
    return out, log[0]


def test_mean_collapses_every_token_to_one_row() -> None:
    t = torch.randn(1, 6, 4)
    out, entry = apply("mean", t)
    # all six token rows identical, and equal to the per-channel mean
    assert torch.allclose(out[0], out[0, :1].expand(6, 4))
    torch.testing.assert_close(out[0, 0], t[0].mean(dim=0))
    assert entry["intervention"] == "mean"
    assert entry["direction_seed"] is None, "mean must not draw from the RNG"


def test_mean_is_deterministic_across_direction_seeds() -> None:
    t = torch.randn(1, 6, 4)
    a, _ = apply("mean", t, base_seed=1)
    b, _ = apply("mean", t, base_seed=999)
    torch.testing.assert_close(a, b)


def test_gauss_matches_the_region_moments_but_destroys_token_structure() -> None:
    t = torch.randn(1, 4096, 8) * 3.0 + 5.0
    out, entry = apply("gauss", t)
    torch.testing.assert_close(out[0].mean(dim=0), t[0].mean(dim=0),
                               rtol=0, atol=0.2)
    torch.testing.assert_close(out[0].std(dim=0), t[0].std(dim=0),
                               rtol=0, atol=0.2)
    assert not torch.allclose(out, t)
    assert entry["direction_seed"] is not None, "gauss must record its seed"


def test_additive_preserves_the_original_signal() -> None:
    t = torch.randn(1, 64, 8)
    out, entry = apply("additive", t, lam=1.0)
    # the perturbed tensor still correlates with the original: noise is added,
    # not substituted
    r = torch.corrcoef(torch.stack([t.flatten(), out.flatten()]))[0, 1]
    assert r > 0.5
    assert entry["intervention"] == "additive"
    assert entry["delta_norm"] == pytest.approx(
        float((out - t).norm()), rel=1e-5)


def test_additive_direction_is_unchanged_by_the_new_code_path() -> None:
    """Phases A-A3 must stay bit-reproducible after adding mean/gauss.

    The direction depends only on (base seed, key), so a fixed pair must still
    give the exact seed those runs recorded.
    """
    assert MODULE.direction_seed(5, "s_prot", False) == (
        5 * 1000003 + __import__("zlib").crc32(b"s_prot")) % (2**31 - 1)
    t = torch.arange(96, dtype=torch.float32).reshape(1, 12, 8)
    a, ea = apply("additive", t, lam=0.6, key="s_lig", base_seed=3)
    b, eb = apply("additive", t, lam=0.6, key="s_lig", base_seed=3)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert ea["direction_seed"] == eb["direction_seed"]


def test_channel_moments_collapses_pair_blocks_too() -> None:
    """z regions are (batch, i, j, channels); both token axes must collapse."""
    z = torch.randn(1, 5, 7, 3)
    mu, sd = MODULE.channel_moments(z)
    assert mu.shape == (1, 1, 1, 3)
    torch.testing.assert_close(mu[0, 0, 0], z[0].reshape(-1, 3).mean(dim=0))
    assert sd.shape == (1, 1, 1, 3)


def test_s_all_cannot_be_combined_with_its_own_subsets() -> None:
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--regions", "s_all,s_lig",
         "--intervention", "mean", "--lam", "0",
         "--out-dir", "/dev/null", "--log", "/dev/null"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "already covers every s row" in r.stderr


def test_destructive_interventions_reject_a_lambda() -> None:
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--regions", "s_all",
         "--intervention", "mean", "--lam", "1.0",
         "--out-dir", "/dev/null", "--log", "/dev/null"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "no scale parameter" in r.stderr


# --------------------------------------------------------------------- shuffle
# Added for exp097 phase A5 / exp099's shuffle arm. `mean` and `gauss` both
# destroy token identity, but they also leave the data manifold: `mean` makes the
# block rank-1 and `gauss` replaces real rows with synthetic draws. `shuffle`
# destroys the same information while every row stays a row the trunk produced.
# These tests pin the three properties that claim rests on.

def test_shuffle_preserves_the_row_multiset_exactly() -> None:
    """Every output row is an input row, and each is used once."""
    t = torch.randn(1, 12, 5)
    out, entry = apply("shuffle", t)
    before = sorted(tuple(r) for r in t[0].tolist())
    after = sorted(tuple(r) for r in out[0].tolist())
    assert before == after
    assert entry["intervention"] == "shuffle"
    assert sorted(entry["permutation"]) == list(range(12))


def test_shuffle_preserves_norm_mean_and_std_exactly() -> None:
    """A permutation is an isometry: these are exact, not approximate."""
    t = torch.randn(1, 33, 384) * 2.0 + torch.linspace(-5, 5, 384)
    out, entry = apply("shuffle", t)
    torch.testing.assert_close(out.norm(), t.norm())
    torch.testing.assert_close(out[0].mean(dim=0), t[0].mean(dim=0))
    torch.testing.assert_close(out[0].std(dim=0), t[0].std(dim=0))
    assert entry["norm_preserved"] is True
    assert entry["base_norm"] == pytest.approx(entry["new_norm"], rel=1e-5)


def test_shuffle_keeps_the_block_full_rank_where_mean_does_not() -> None:
    """The point of the arm: same information destroyed, form intact."""
    t = torch.randn(1, 20, 6)
    shuffled, _ = apply("shuffle", t)
    meaned, _ = apply("mean", t)
    assert torch.linalg.matrix_rank(shuffled[0]).item() == \
           torch.linalg.matrix_rank(t[0]).item()
    assert torch.linalg.matrix_rank(meaned[0]).item() == 1


def test_shuffle_is_a_derangement_so_no_token_keeps_its_row() -> None:
    t = torch.randn(1, 33, 8)
    for seed in (1, 2, 3, 17, 404):
        out, entry = apply("shuffle", t, base_seed=seed)
        perm = torch.tensor(entry["permutation"])
        assert int((perm == torch.arange(33)).sum()) == 0
        assert entry["rows_moved"] == 33
        assert entry["n_rows"] == 33
        # and it genuinely moved the data, not just the index vector
        assert not torch.allclose(out, t)


def test_shuffle_is_reproducible_from_its_seed_and_varies_with_it() -> None:
    t = torch.randn(1, 33, 8)
    a, ea = apply("shuffle", t, base_seed=5)
    b, eb = apply("shuffle", t, base_seed=5)
    c, ec = apply("shuffle", t, base_seed=6)
    torch.testing.assert_close(a, b)
    assert ea["permutation"] == eb["permutation"]
    assert ea["permutation"] != ec["permutation"]
    assert ea["direction_seed"] is not None, "shuffle must record its seed"


def test_shuffle_refuses_a_pair_block_because_there_is_no_row_axis() -> None:
    """A z block is indexed by a token PAIR; permuting 'rows' is undefined."""
    t = torch.randn(1, 6, 6, 4)
    with pytest.raises(SystemExit, match="no single row axis"):
        apply("shuffle", t, key="z_lig")


def test_shuffle_is_a_noop_on_a_single_row_rather_than_an_error() -> None:
    """A one-token region has nothing to permute; it must not raise."""
    t = torch.randn(1, 1, 4)
    out, entry = apply("shuffle", t)
    torch.testing.assert_close(out, t)
    assert entry["rows_moved"] == 0


# --- phase A7: the conditioning-level hook ------------------------------------

def test_bind_args_accepts_keywords():
    names = ("s_trunk", "z_trunk", "rpe", "feats")
    out = MODULE.bind_args(names, (), {"s_trunk": 1, "z_trunk": 2, "rpe": 3, "feats": 4})
    assert out == {"s_trunk": 1, "z_trunk": 2, "rpe": 3, "feats": 4}


def test_bind_args_accepts_positionals():
    """The training checkpoint branch calls DiffusionConditioning positionally."""
    names = ("s_trunk", "z_trunk", "rpe", "feats")
    assert MODULE.bind_args(names, (1, 2, 3, 4), {}) == {
        "s_trunk": 1, "z_trunk": 2, "rpe": 3, "feats": 4}


def test_bind_args_accepts_a_mix():
    names = ("s_trunk", "z_trunk", "rpe", "feats")
    assert MODULE.bind_args(names, (1, 2), {"rpe": 3, "feats": 4}) == {
        "s_trunk": 1, "z_trunk": 2, "rpe": 3, "feats": 4}


def test_bind_args_rejects_a_duplicate():
    """Silently preferring one would perturb the wrong tensor."""
    with pytest.raises(SystemExit):
        MODULE.bind_args(("a", "b"), (1,), {"a": 9, "b": 2})


def test_bind_args_rejects_a_missing_argument():
    with pytest.raises(SystemExit):
        MODULE.bind_args(("a", "b"), (1,), {})


def test_bind_args_rejects_too_many_positionals():
    with pytest.raises(SystemExit):
        MODULE.bind_args(("a", "b"), (1, 2, 3), {})


def test_hook_choices_include_the_complete_one():
    assert MODULE.HOOKS == ("sample", "conditioning", "both")


# --- phase A9: restricting an intervention to sigma-ranked channels -----------

def _block():
    """4 tokens x 5 channels with deliberately ordered across-token spread."""
    return torch.tensor([[[0., 0., 0., 0., 0.],
                          [1., 10., 100., 1000., 10000.],
                          [2., 20., 200., 2000., 20000.],
                          [3., 30., 300., 3000., 30000.]]])


def test_sigma_ranking_picks_the_widest_channels():
    sel = MODULE.sigma_ranked_channels(_block(), 2, "top")
    assert sel.tolist() == [3, 4]


def test_sigma_ranking_bottom_picks_the_narrowest():
    sel = MODULE.sigma_ranked_channels(_block(), 2, "bottom")
    assert sel.tolist() == [0, 1]


def test_sigma_ranking_rejects_an_out_of_range_count():
    for n in (0, 6, -1):
        with pytest.raises(SystemExit):
            MODULE.sigma_ranked_channels(_block(), n, "top")


def test_channel_restriction_leaves_other_channels_untouched():
    t = _block().clone()
    new, entry = apply("gauss", t, key="s_lig", n_channels=2)
    assert entry["n_channels"] == 2 and entry["channels"] == [3, 4]
    before = _block()
    assert torch.equal(new[..., :3], before[..., :3])      # untouched
    assert not torch.equal(new[..., 3:], before[..., 3:])  # resampled


def test_all_channels_matches_the_unrestricted_arm():
    """n = every channel must reproduce the plain intervention exactly."""
    a, _ = apply("gauss", _block().clone(), key="s_lig")
    b, _ = apply("gauss", _block().clone(), key="s_lig", n_channels=5)
    assert torch.equal(a, b)
