#!/usr/bin/env python3
"""Perturb any subset of the five conditioning quadrants Boltz-2 hands to diffusion.

exp065 perturbed whole tensors; exp068 split them by molecule but drew one noise
direction per region and reused it at every lambda, so a rescue cannot be told
apart from a lucky direction. This runner parameterizes the direction so the same
(region, lambda) can be sampled many times, and admits arbitrary region sets so
the quadrants and their combinations come from one code path.

Token order is protein first then ligand (mol_type 0 for indices 0-361, mol_type 3
for 362-394 on 8SGE), so every region is a contiguous slice:

  s_prot  s_trunk rows of the protein tokens
  s_lig   s_trunk rows of the ligand tokens
  z_pp    the protein-protein block of the token attention bias
  z_pl    both protein-ligand cross blocks
  z_lig   the ligand-ligand block
  s_atoms s_trunk rows of an explicit token list (--token-subset), for probing a
          named handful of ligand atoms instead of the whole ligand. Unlike the
          five quadrants this one is not a contiguous slice and not a partition
          member; it is a subset of s_lig.
  s_all   every s_trunk row, protein and ligand together. Not a partition member
          either: it is the union of s_prot and s_lig, and exists so the whole
          single representation can be ablated as one stream the way Park et al.
          do, rather than molecule by molecule.

Three interventions are available (--intervention), because "perturb" and
"ablate" answer different questions:

  additive  (default) add lambda * sigma(region) * eps. The original signal is
            still present, buried under noise. This is what phases A-A3 used and
            the only intervention for which --lam is meaningful.
  mean      replace every row in the region with the region's per-channel mean,
            destroying all token-to-token variation while preserving the channel
            means. This is Park et al.'s averaging ablation (their Sec. 3.1).
            Deterministic: no direction seed is drawn.
  gauss     replace every row with an independent Gaussian sample matched to the
            region's per-channel mean and standard deviation. Park et al.'s
            second ablation. Destroys token-to-token structure like `mean` but
            keeps per-channel spread, so a result that holds under both is not an
            artifact of collapsing the region to a single point.

Averaging over a region that spans two molecule types gives ligand rows the
protein-dominated complex mean, which is out of distribution. That is Park et
al.'s own construction and is why `s_prot,s_lig` (each molecule averaged within
itself) is run alongside `s_all` as the less out-of-distribution comparison.
Zeroing is deliberately not offered: the paper declines it as evidence because it
changes the representation's scale as well as its content.

Noise is scaled to the standard deviation of the perturbed region itself, so a
given lambda means the same relative displacement whichever region is chosen. The
direction never touches the model RNG stream, so the diffusion trajectory is
identical across scales apart from the conditioning change.

Note what is actually perturbed: s_trunk, and the token_trans_bias that the
diffusion module consumes. The pair representation reaches diffusion only as an
additive pair bias, so that bias is where a z-side intervention has to land.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
# The checkpoint and the Boltz cache are machine-local and large (the cache is
# several gigabytes), so they are never vendored. Both are named by flag, and
# both fall back to an environment variable so a shell can set them once.
DEFAULT_CHECKPOINT = os.environ.get("BOLTZ_CHECKPOINT", "")
DEFAULT_CACHE = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))


class _BlockDumped(Exception):
    """Raised to stop a run once the requested block has been written.

    --dump-block-only exists to recover a trunk state cheaply. The block is
    complete the moment the trunk hands off to diffusion, so the 200 denoising
    steps that follow are pure waste when all we want is `s`. Raising here
    unwinds out of boltz's predict loop; main() catches this one exception by
    identity and records the run as a dump rather than a failure.
    """


ATOMIC_REGIONS = ("s_prot", "s_lig", "z_pp", "z_pl", "z_lig", "s_atoms",
                  "s_tokens", "s_all")
INTERVENTIONS = ("additive", "mean", "gauss", "shuffle")
S_REGIONS = ("s_prot", "s_lig", "s_atoms", "s_tokens", "s_all")


def parse_regions(spec):
    """Comma-separated atomic region names, order preserved, duplicates dropped."""
    want = [r.strip() for r in spec.split(",") if r.strip()]
    unknown = [r for r in want if r not in ATOMIC_REGIONS]
    if unknown:
        raise SystemExit(f"unknown region(s) {unknown}; choose from {list(ATOMIC_REGIONS)}")
    if not want:
        raise SystemExit("--regions must name at least one region, or use --lam 0")
    return tuple(dict.fromkeys(want))


def parse_token_subset(spec):
    """Comma-separated absolute token indices, sorted, duplicates dropped."""
    if not spec.strip():
        return ()
    try:
        idx = sorted({int(t) for t in spec.split(",") if t.strip()})
    except ValueError:
        raise SystemExit(f"--token-subset must be integers, got {spec!r}")
    if any(i < 0 for i in idx):
        raise SystemExit("--token-subset indices must be non-negative")
    return tuple(idx)


def ligand_span(feats):
    """Return (start, end) token indices of the ligand, asserting contiguity."""
    mol = feats["mol_type"][0]
    idx = torch.nonzero(mol != 0, as_tuple=False).flatten()
    if idx.numel() == 0:
        raise RuntimeError("no non-protein tokens found")
    lo, hi = int(idx.min()), int(idx.max()) + 1
    if hi - lo != idx.numel():
        raise RuntimeError("ligand tokens are not contiguous; regional slicing invalid")
    return lo, hi


def direction_seed(base, key, legacy):
    """Per-region seed.

    exp068 seeded one generator with a single constant for every region, so two
    regions of equal shape received an identical direction. --legacy-direction
    reproduces that exactly; otherwise each region draws independently.
    """
    if legacy:
        return int(base)
    return (int(base) * 1000003 + zlib.crc32(key.encode())) % (2**31 - 1)


def channel_moments(sub):
    """Per-channel mean and std over every non-channel axis except the batch.

    `sub` is (batch, tokens, channels) for an s region or (batch, i, j, channels)
    for a z block; in both cases the last axis is the feature channel and the
    axes between carry the tokens being collapsed. Averaging here is therefore
    "the region's mean row", which is the quantity Park et al. average.
    """
    axes = tuple(range(1, sub.dim() - 1))
    f = sub.float()
    return f.mean(dim=axes, keepdim=True), f.std(dim=axes, keepdim=True)


HOOKS = ("sample", "conditioning", "both")


def bind_args(names, args, kwargs):
    """Normalise a mixed positional/keyword call into one kwargs dict.

    DiffusionConditioning is called with keywords at inference and positionally
    from the training checkpoint branch. Wrapping it has to survive both, and
    guessing wrong would silently perturb `relative_position_encoding` instead of
    `z_trunk`, which would look like a result.
    """
    if len(args) > len(names):
        raise SystemExit(f"expected at most {len(names)} positional arguments, "
                         f"got {len(args)}")
    bound = dict(zip(names, args))
    overlap = bound.keys() & kwargs.keys()
    if overlap:
        raise SystemExit(f"argument(s) {sorted(overlap)} given both positionally "
                         f"and by keyword")
    bound.update(kwargs)
    missing = [n for n in names if n not in bound]
    if missing:
        raise SystemExit(f"missing argument(s) {missing}")
    return bound


def sigma_ranked_channels(sub, n, order):
    """The n channels whose values vary most (or least) across the region's tokens.

    Ranked at RUNTIME from the block the hook is holding, not from a precomputed
    list, so the same flag means the same thing at any seed. The ranking is a
    property of the trained model rather than of one run -- across 31 independent
    conditionings its rank correlation is 0.979 and channel 100 is top in all of
    them -- but recomputing it costs nothing and removes a whole class of
    stale-constant bug.

    Sigma here is the spread ACROSS TOKENS, which is what distinguishes one ligand
    atom from another. A channel giving all 33 atoms nearly the same value carries
    no per-atom identity however large its values are.
    """
    axes = tuple(range(1, sub.dim() - 1))
    sd = sub.float().std(dim=axes).flatten()          # (channels,)
    if not 0 < n <= sd.numel():
        raise SystemExit(f"--n-channels {n} outside 1..{sd.numel()}")
    idx = sd.argsort(descending=(order == "top"))[:n]
    return idx.sort().values


def install_tracer(store):
    """Record the denoiser's running clean-structure estimate at every step.

    Patches a DIFFERENT method from install(): the perturbation wraps
    AtomDiffusion.sample, this wraps AtomDiffusion.preconditioned_network_forward,
    so the two compose and a perturbed run can be traced. Lifted verbatim in
    behaviour from experiments/exp067_xhat0_trajectory/trace_xhat0.py so the
    arrays this writes are byte-compatible with exp067's analysis scripts.

    x0-hat is the model's current best guess at the finished structure, so frame 0
    is what the model predicts having seen nothing but noise. The sampler applies a
    random rigid-body augmentation before each step, so the frames do not share a
    coordinate frame; anything read off them must be either frame-invariant (a
    dihedral, a radius of gyration) or scored after independent superposition.
    """
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    original = AtomDiffusion.preconditioned_network_forward

    def patched(self, noised_atom_coords, sigma, network_condition_kwargs):
        out = original(self, noised_atom_coords, sigma, network_condition_kwargs)
        store["xhat0"].append(out.detach().float().cpu().numpy()[0].copy())
        store["x"].append(noised_atom_coords.detach().float().cpu().numpy()[0].copy())
        store["sigma"].append(float(sigma) if not torch.is_tensor(sigma)
                              else float(sigma.flatten()[0]))
        return out

    AtomDiffusion.preconditioned_network_forward = patched
    return original


def install(regions, lam, base_seed, legacy, log, token_subset=(),
            intervention="additive", dump_dir=None, dump_only=False,
            hook="sample", n_channels=0, channel_order="top"):
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    original = AtomDiffusion.sample
    cache = {}

    def direction(like, key):
        if key not in cache:
            seed = direction_seed(base_seed, key, legacy)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            cache[key] = (torch.randn(like.shape, generator=gen, dtype=torch.float32)
                          .to(like.device), seed)
        tensor, seed = cache[key]
        return tensor.to(dtype=like.dtype), seed

    def permutation(n, key):
        """A derangement of n row indices: no row keeps its own position.

        Drawn by rejection from uniform permutations, which needs ~e draws. Seeded
        from the same machinery as `direction`, so a replicate index reproduces its
        permutation exactly.
        """
        seed = direction_seed(base_seed, key, legacy)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        identity = torch.arange(n)
        for _ in range(10000):
            perm = torch.randperm(n, generator=gen)
            if n < 2 or int((perm == identity).sum()) == 0:
                return perm, seed
        raise SystemExit(f"could not draw a derangement of {n} rows")

    def bump(tensor, sl, key, label):
        """Apply the chosen intervention in place over the given slice."""
        sub = tensor[sl]
        if dump_dir is not None:
            # The UNPERTURBED block, saved before anything is written to it. The
            # per-channel mean and std the interventions are built from are
            # functions of this tensor alone, so dumping it once per conditioning
            # lets an offline analysis recompute every intervention's geometry
            # exactly instead of estimating it from the scalar norms logged below.
            # A raw model internal: it belongs in runs/, never in the repository.
            out = Path(dump_dir)
            out.mkdir(parents=True, exist_ok=True)
            name = key.replace(":", "_").replace(",", "-")
            torch.save(sub.detach().float().cpu().clone(), out / f"{name}.pt")
            if dump_only:
                raise _BlockDumped(name)
        base_norm = float(sub.float().norm())
        sigma = float(sub.float().std())
        seed = None

        if intervention == "additive":
            eps, seed = direction(sub, key)
            delta = lam * sigma * eps
            new = sub + delta
        elif intervention == "mean":
            # every row becomes the region's mean row; deterministic, no RNG
            mu, _ = channel_moments(sub)
            new = mu.expand_as(sub).to(dtype=sub.dtype)
        elif intervention == "gauss":
            mu, sd = channel_moments(sub)
            eps, seed = direction(sub, key)
            new = (mu + sd * eps.float()).to(dtype=sub.dtype)
        elif intervention == "shuffle":
            # Permute the token rows among themselves. This destroys which token
            # owns which row just as completely as `mean` or `gauss` do, but every
            # row remains a row the trunk actually produced: the block stays full
            # rank, the per-channel mean and standard deviation are preserved
            # EXACTLY rather than approximately, and the Frobenius norm does not
            # change at all. It is therefore the on-manifold counterpart of those
            # two, and the arm that separates "the information was lost" from "the
            # representation's form was broken".
            if sub.dim() != 3:
                raise SystemExit(
                    f"shuffle permutes token ROWS and is defined for the s regions, "
                    f"whose block is (batch, tokens, channels); got {sub.dim()} "
                    f"dimensions for {label}. A z block is indexed by a token PAIR, "
                    f"so there is no single row axis to permute.")
            perm, seed = permutation(sub.shape[1], key)
            new = sub.index_select(1, perm.to(sub.device))
            moved = int((perm != torch.arange(len(perm))).sum())
        else:                                     # unreachable: validated in main
            raise SystemExit(f"unknown intervention {intervention!r}")

        if n_channels:
            # restrict the intervention to a subset of CHANNELS: every other
            # channel keeps the value the trunk produced. Applied after `new` is
            # built so the moments are still taken over the whole block, which is
            # what makes the n = all case identical to the unrestricted arm.
            chans = sigma_ranked_channels(sub, n_channels, channel_order)
            restricted = sub.clone()
            restricted[..., chans] = new[..., chans]
            new = restricted
            sel_channels = chans.tolist()
        else:
            sel_channels = None

        # tensor[sl] is a VIEW, so the write below mutates `sub` too; measure the
        # change first or delta_norm reads zero
        delta_norm = float((new - sub).float().norm())
        new_norm = float(new.float().norm())
        tensor[sl] = new
        entry = {"region": label, "key": key, "intervention": intervention,
                 "direction_seed": seed,
                 "shape": list(sub.shape), "elements": int(sub.numel()),
                 "std": sigma,
                 "delta_norm": delta_norm,
                 "base_norm": base_norm,
                 "new_norm": new_norm}
        if sel_channels is not None:
            entry["n_channels"] = len(sel_channels)
            entry["channel_order"] = channel_order
            entry["channels"] = sel_channels
        if intervention == "shuffle":
            # a permutation preserves the norm exactly; recording both makes that
            # checkable from the log rather than taken on trust
            entry["rows_moved"] = moved
            entry["n_rows"] = int(sub.shape[1])
            entry["permutation"] = perm.tolist()
            entry["norm_preserved"] = abs(new_norm - base_norm) < 1e-3 * base_norm
        log.append(entry)

    def patched(self, atom_mask, num_sampling_steps=None, **kwargs):
        # `additive` with lam 0 is a no-op, so it stays gated on lam; the
        # destructive interventions have no scale parameter and fire whenever a
        # region is named.
        active = bool(regions) and (intervention != "additive" or lam > 0.0)
        if active:
            lo, hi = ligand_span(kwargs["feats"])
            if set(S_REGIONS) & set(regions):
                s = kwargs["s_trunk"].clone()
                if "s_all" in regions:
                    n_tok = s.shape[1]
                    if n_tok != hi:
                        raise SystemExit(
                            f"s_trunk has {n_tok} token rows but the ligand ends at "
                            f"{hi}; s_all would average over padding")
                    bump(s, (slice(None), slice(0, n_tok), slice(None)),
                         "s_all", "s_trunk[all]")
                if "s_prot" in regions:
                    bump(s, (slice(None), slice(0, lo), slice(None)),
                         "s_prot", "s_trunk[protein]")
                if "s_lig" in regions:
                    bump(s, (slice(None), slice(lo, hi), slice(None)),
                         "s_lig", "s_trunk[ligand]")
                if "s_atoms" in regions:
                    if not token_subset:
                        raise SystemExit("region s_atoms requires --token-subset")
                    out = [i for i in token_subset if not lo <= i < hi]
                    if out:
                        raise SystemExit(
                            f"--token-subset {out} outside the ligand span "
                            f"[{lo}, {hi}); s_atoms is a subset of s_lig")
                    sel = torch.tensor(token_subset, device=s.device, dtype=torch.long)
                    # the key carries the subset, so different subsets draw
                    # independent directions for the same --direction-seed
                    key = "s_atoms:" + ",".join(str(i) for i in token_subset)
                    bump(s, (slice(None), sel, slice(None)), key,
                         f"s_trunk[tokens {list(token_subset)}]")
                if "s_tokens" in regions:
                    # An ARBITRARY token subset, with no containment requirement.
                    # s_atoms deliberately refuses tokens outside the ligand span,
                    # which is the right guard for its purpose; this region exists
                    # for the size-matched control that has to sit OUTSIDE it --
                    # destroy as many protein tokens as the ligand has, far from
                    # the site, and see whether the torsion still releases.
                    if not token_subset:
                        raise SystemExit("region s_tokens requires --token-subset")
                    n_tok = s.shape[1]
                    bad = [i for i in token_subset if not 0 <= i < n_tok]
                    if bad:
                        raise SystemExit(f"--token-subset {bad} outside the "
                                         f"{n_tok} available tokens")
                    sel = torch.tensor(token_subset, device=s.device,
                                       dtype=torch.long)
                    key = "s_tokens:" + ",".join(str(i) for i in token_subset)
                    bump(s, (slice(None), sel, slice(None)), key,
                         f"s_trunk[{len(token_subset)} tokens]")
                kwargs["s_trunk"] = s
            if {"z_pp", "z_pl", "z_lig"} & set(regions):
                dc = dict(kwargs["diffusion_conditioning"])
                b = dc["token_trans_bias"].clone()
                if "z_pp" in regions:
                    bump(b, (slice(None), slice(0, lo), slice(0, lo), slice(None)),
                         "z_pp", "z[protein,protein]")
                if "z_pl" in regions:
                    bump(b, (slice(None), slice(0, lo), slice(lo, hi), slice(None)),
                         "z_pl_a", "z[protein,ligand]")
                    bump(b, (slice(None), slice(lo, hi), slice(0, lo), slice(None)),
                         "z_pl_b", "z[ligand,protein]")
                if "z_lig" in regions:
                    bump(b, (slice(None), slice(lo, hi), slice(lo, hi), slice(None)),
                         "z_lig", "z[ligand,ligand]")
                dc["token_trans_bias"] = b
                kwargs["diffusion_conditioning"] = dc
        return original(self, atom_mask, num_sampling_steps=num_sampling_steps, **kwargs)

    def patched_conditioning(self, *args, **kwargs):
        """Perturb s_trunk / z_trunk BEFORE the diffusion conditioning is built.

        boltz2.py:517 turns z_trunk into five things the diffusion module uses --
        q, c, atom_enc_bias, atom_dec_bias and token_trans_bias -- and then calls
        sample(). The `sample` hook can only reach token_trans_bias, and only
        after the other four were computed from the untouched z. For z this hook
        is COMPLETE, because z reaches diffusion through no other route. For s it
        is not: s_trunk is also handed to sample() directly, so a complete s
        perturbation needs --hook both.
        """
        names = ("s_trunk", "z_trunk", "relative_position_encoding", "feats")
        kw = bind_args(names, args, kwargs)
        active = bool(regions) and (intervention != "additive" or lam > 0.0)
        if active:
            lo, hi = ligand_span(kw["feats"])
            if set(S_REGIONS) & set(regions):
                st = kw["s_trunk"].clone()
                n_tok = st.shape[1]
                if "s_all" in regions:
                    if n_tok != hi:
                        raise SystemExit(
                            f"s_trunk has {n_tok} token rows but the ligand ends "
                            f"at {hi}; s_all would average over padding")
                    bump(st, (slice(None), slice(0, n_tok), slice(None)),
                         "s_all", "cond:s_trunk[all]")
                if "s_prot" in regions:
                    bump(st, (slice(None), slice(0, lo), slice(None)),
                         "s_prot", "cond:s_trunk[protein]")
                if "s_lig" in regions:
                    bump(st, (slice(None), slice(lo, hi), slice(None)),
                         "s_lig", "cond:s_trunk[ligand]")
                if "s_atoms" in regions:
                    if not token_subset:
                        raise SystemExit("region s_atoms requires --token-subset")
                    sel = torch.tensor(token_subset, device=st.device,
                                       dtype=torch.long)
                    key = "s_atoms:" + ",".join(str(i) for i in token_subset)
                    bump(st, (slice(None), sel, slice(None)), key,
                         f"cond:s_trunk[tokens {list(token_subset)}]")
                if "s_tokens" in regions:
                    # An ARBITRARY token subset, with no containment requirement.
                    # s_atoms deliberately refuses tokens outside the ligand span,
                    # which is the right guard for its purpose; this region exists
                    # for the size-matched control that has to sit OUTSIDE it --
                    # destroy as many protein tokens as the ligand has, far from
                    # the site, and see whether the torsion still releases.
                    if not token_subset:
                        raise SystemExit("region s_tokens requires --token-subset")
                    n_tok = st.shape[1]
                    bad = [i for i in token_subset if not 0 <= i < n_tok]
                    if bad:
                        raise SystemExit(f"--token-subset {bad} outside the "
                                         f"{n_tok} available tokens")
                    sel = torch.tensor(token_subset, device=st.device,
                                       dtype=torch.long)
                    key = "s_tokens:" + ",".join(str(i) for i in token_subset)
                    bump(st, (slice(None), sel, slice(None)), key,
                         f"cond:s_trunk[{len(token_subset)} tokens]")
                kw["s_trunk"] = st
            if {"z_pp", "z_pl", "z_lig"} & set(regions):
                zt = kw["z_trunk"].clone()
                if "z_pp" in regions:
                    bump(zt, (slice(None), slice(0, lo), slice(0, lo), slice(None)),
                         "z_pp", "cond:z_trunk[protein,protein]")
                if "z_pl" in regions:
                    bump(zt, (slice(None), slice(0, lo), slice(lo, hi), slice(None)),
                         "z_pl_a", "cond:z_trunk[protein,ligand]")
                    bump(zt, (slice(None), slice(lo, hi), slice(0, lo), slice(None)),
                         "z_pl_b", "cond:z_trunk[ligand,protein]")
                if "z_lig" in regions:
                    bump(zt, (slice(None), slice(lo, hi), slice(lo, hi), slice(None)),
                         "z_lig", "cond:z_trunk[ligand,ligand]")
                kw["z_trunk"] = zt
        return original_cond(self, **kw)

    original_cond = None
    if hook in ("conditioning", "both"):
        from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
        original_cond = DiffusionConditioning.forward
        DiffusionConditioning.forward = patched_conditioning
    if hook in ("sample", "both"):
        AtomDiffusion.sample = patched
    return original


def install_msa_recorder(log: list):
    """Record which MSA rows each recycle actually subsampled. Read-only.

    boltz/model/modules/trunkv2.py:632 draws `torch.randperm(depth)[:k]` off the
    global RNG inside MSAModule.forward, so the subsample is a function of --seed
    and is REDRAWN every recycle. (trunkv2 is the Boltz-2 trunk; trunk.py holds the
    Boltz-1 one, which does the same thing at line 267 but is not what runs here.) The indices are local to that function and never
    returned, so they are recovered here by replay: capture the RNG state before
    the call, let the module run untouched, then rewind to the captured state and
    repeat the same draw. The randperm is the first RNG consumption in that
    forward (the module is in eval mode, so no dropout precedes it), which is what
    makes the replay exact rather than merely similar.

    The post-call RNG state is restored afterwards, so the replay cannot perturb
    anything downstream and the prediction is bit-identical to an unhooked run.
    """
    from boltz.model.modules.trunkv2 import MSAModule

    original = MSAModule.forward
    state = {"calls": 0}

    def patched(self, z, emb, feats, *a, **kw):
        depth = int(feats["msa"].shape[1])
        before = torch.get_rng_state()
        out = original(self, z, emb, feats, *a, **kw)
        after = torch.get_rng_state()
        state["calls"] += 1
        entry = {"call": state["calls"], "msa_depth": depth,
                 "subsample": bool(self.subsample_msa),
                 "num_subsampled": int(self.num_subsampled_msa)
                                   if self.subsample_msa else None}
        if self.subsample_msa:
            torch.set_rng_state(before)
            idx = torch.randperm(depth)[: self.num_subsampled_msa]
            torch.set_rng_state(after)
            entry["indices"] = sorted(int(i) for i in idx)
            entry["query_row_included"] = 0 in entry["indices"]
        log.append(entry)
        return out

    MSAModule.forward = patched
    return state


def checkpoint_sha256(path):
    """SHA-256 of the checkpoint actually used, cached beside it.

    This used to be a hardcoded constant, which silently recorded the wrong
    provenance for anyone running a different checkpoint. Hashing a multi-
    gigabyte file on every run is wasteful, so the digest is written to a
    sidecar and reused while the checkpoint's size and mtime are unchanged.
    """
    if not path:
        return "unavailable"
    target = Path(path)
    if not target.is_file():
        return "unavailable"
    stat = target.stat()
    stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
    sidecar = target.with_suffix(target.suffix + ".sha256")
    try:
        cached_stamp, cached_digest = sidecar.read_text().split(None, 1)
        if cached_stamp == stamp:
            return cached_digest.strip()
    except Exception:
        pass
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    try:
        sidecar.write_text(f"{stamp} {value}\n")
    except OSError:
        pass          # a read-only checkpoint directory is not a failure
    return value


def git_revision():
    """The revision this run came from, or "unavailable" outside a work tree.

    Provenance is recorded on every run, but a user who downloaded a tarball
    rather than cloning should still be able to run it.
    """
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="",
                    help="comma-separated subset of " + ",".join(ATOMIC_REGIONS))
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="Boltz-2 .ckpt; defaults to $BOLTZ_CHECKPOINT")
    ap.add_argument("--cache", default=DEFAULT_CACHE,
                    help="Boltz cache directory; defaults to $BOLTZ_CACHE or ~/.boltz")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42, help="model/diffusion seed")
    ap.add_argument("--direction-seed", type=int, default=12345,
                    help="seed for the perturbation direction, independent of --seed")
    ap.add_argument("--token-subset", default="",
                    help="absolute token indices for region s_atoms, e.g. 369,370,371,381")
    ap.add_argument("--intervention", default="additive", choices=INTERVENTIONS,
                    help="additive (lambda-scaled noise, the default) or the "
                         "destructive ablations mean/gauss, which ignore --lam")
    ap.add_argument("--legacy-direction", action="store_true",
                    help="reproduce exp068: one shared generator seed for every region")
    ap.add_argument("--trace", default=None,
                    help="npz path for the per-step x0-hat trajectory; records every "
                         "denoiser call. Same arrays as exp067_xhat0_trajectory "
                         "(xhat0, x, sigma), so exp067's analysis reads them directly. "
                         "Raw model internals: write under runs/, never the repository.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--inputs", default=str(ROOT / "runs" / "inputs"),
                    help="directory of Boltz input YAMLs; scripts/run_8sge.py "
                         "materialises the vendored 8SGE system there")
    ap.add_argument("--experiment-id", default="8sge_slig_rescue",
                    help="experiment id written into the run log; other experiments "
                         "that reuse this runner (exp104 8CJG) pass their own")
    ap.add_argument("--dump-msa", default=None,
                    help="write a JSON record of which MSA rows each recycle "
                         "subsampled; read-only, the prediction is unchanged")
    ap.add_argument("--n-channels", type=int, default=0,
                    help="restrict the intervention to this many channels, ranked "
                         "by their spread across the region's tokens; 0 = all")
    ap.add_argument("--channel-order", default="top", choices=("top", "bottom"),
                    help="take the highest-sigma channels (top) or lowest (bottom)")
    ap.add_argument("--hook", default="sample", choices=HOOKS,
                    help="where the intervention is applied. `sample` (default, and "
                         "what every run before phase A7 used) patches "
                         "AtomDiffusion.sample and can only reach token_trans_bias "
                         "for z. `conditioning` patches DiffusionConditioning.forward, "
                         "which is COMPLETE for z. `both` is needed for a complete s "
                         "perturbation, since s_trunk also goes straight to sample().")
    ap.add_argument("--dump-block-only", action="store_true",
                    help="stop the run as soon as --dump-block has written; skips "
                         "diffusion, so no pose is produced")
    ap.add_argument("--dump-block", default=None,
                    help="directory to save each intervened region's UNPERTURBED "
                         "block to, as <key>.pt, before the intervention is "
                         "applied; a diagnostic, it does not change the run")
    args = ap.parse_args()

    regions = parse_regions(args.regions) if args.regions else ()
    if args.lam > 0.0 and not regions:
        raise SystemExit("--lam > 0 requires --regions")
    token_subset = parse_token_subset(args.token_subset)
    needs_subset = {"s_atoms", "s_tokens"} & set(regions)
    if needs_subset and not token_subset:
        raise SystemExit(f"region(s) {sorted(needs_subset)} require --token-subset")
    if token_subset and not needs_subset:
        raise SystemExit("--token-subset only applies to regions s_atoms / s_tokens")
    if {"s_atoms", "s_tokens"} <= set(regions):
        raise SystemExit("s_atoms and s_tokens both consume --token-subset; "
                         "use one of them")
    if args.intervention != "additive":
        # lambda scales additive noise and has no meaning for a replacement, so
        # accepting it silently would record a parameter that did nothing.
        if args.lam != 0.0:
            raise SystemExit(f"--intervention {args.intervention} has no scale "
                             f"parameter; pass --lam 0")
        if not regions:
            raise SystemExit(f"--intervention {args.intervention} requires --regions")
    if "s_all" in regions and ({"s_prot", "s_lig", "s_atoms", "s_tokens"}
                               & set(regions)):
        # s_all is the union of the others, so a combination would apply two
        # interventions to the same rows and the second would silently win
        raise SystemExit("region s_all already covers every s row; do not combine "
                         "it with s_prot, s_lig, s_atoms or s_tokens")

    log = []
    install(regions, args.lam, args.direction_seed, args.legacy_direction, log,
            token_subset, args.intervention, args.dump_block, args.dump_block_only,
            args.hook, args.n_channels, args.channel_order)
    trace_store = {"xhat0": [], "x": [], "sigma": []}
    if args.trace:
        install_tracer(trace_store)
    msa_log: list = []
    if args.dump_msa:
        install_msa_recorder(msa_log)

    argv = ["boltz", "predict", args.inputs,
            "--out_dir", args.out_dir, "--cache", args.cache,
            "--model", "boltz2", "--checkpoint", args.checkpoint,
            "--accelerator", "gpu", "--devices", "1",
            "--recycling_steps", "3", "--sampling_steps", "200",
            "--diffusion_samples", "1", "--max_parallel_samples", "1",
            "--step_scale", "1.5", "--output_format", "mmcif",
            "--num_workers", "1", "--seed", str(args.seed),
            "--max_msa_seqs", "8192", "--subsample_msa", "--num_subsampled_msa", "1024",
            "--no_kernels"]
    sys.argv = argv
    started = time.monotonic()
    status = "ok"
    try:
        from boltz.main import cli
        try:
            cli.main(args=argv[1:], standalone_mode=False)
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    except _BlockDumped as exc:                   # requested stop, not a failure
        status = f"dumped {exc} and stopped before diffusion"
    except Exception as exc:                      # record the failure, do not hide it
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        runtime = time.monotonic() - started
        peak = (torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available() else None)
        trace_steps = len(trace_store["xhat0"])
        if args.trace and trace_steps:
            Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.trace,
                xhat0=np.stack(trace_store["xhat0"]),
                x=np.stack(trace_store["x"]),
                sigma=np.asarray(trace_store["sigma"], dtype=np.float32))
        elif args.trace:
            status = f"{status}; no denoiser calls recorded, trace not written"
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(json.dumps({
            "experiment": args.experiment_id,
            "git_revision": git_revision(),
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
            "regions": list(regions),
            "token_subset": list(token_subset),
            "intervention": args.intervention,
            "hook": args.hook,
            "n_channels": args.n_channels,
            "channel_order": args.channel_order,
            "lam": args.lam,
            "model_seed": args.seed,
            "direction_seed_base": args.direction_seed,
            "legacy_direction": args.legacy_direction,
            "dump_block": args.dump_block,
            "trace": args.trace,
            "trace_steps": trace_steps,
            "command": argv,
            "runtime_seconds": round(runtime, 1),
            "peak_gpu_mib": None if peak is None else round(peak, 1),
            "status": status,
            "perturbations": log,
        }, indent=2) + "\n")
        if args.dump_msa:
            Path(args.dump_msa).parent.mkdir(parents=True, exist_ok=True)
            Path(args.dump_msa).write_text(json.dumps({
                "experiment": args.experiment_id,
                "model_seed": args.seed,
                "inputs": args.inputs,
                "recycling_steps": 3,
                "msa_calls": msa_log,
            }, indent=1) + "\n")
    if args.dump_block_only:
        print(f"[dump-only] {status}")
        return
    # the shared scorer requires a provenance record beside the predictions
    collection = Path(args.out_dir) / "collection.json"
    collection.write_text(json.dumps({
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "layout": Path(args.out_dir).name,
        "poses": [{"pose_index": 0}],
    }, indent=2) + "\n")

    print(f"regions={'+'.join(regions) or 'none'} intervention={args.intervention} "
          f"lam={args.lam} dir_seed={args.direction_seed} blocks={len(log)} "
          f"runtime={runtime:.0f}s status={status}")


if __name__ == "__main__":
    main()
