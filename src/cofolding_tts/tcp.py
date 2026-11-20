"""[reproduce] Boltz-Perturb's Token Conditioning Perturbation (TCP), ported.

WHY A PORT RATHER THAN THEIR FORK
  Boltz-Perturb ships a modified copy of boltz/model/modules/diffusionv2.py. Running
  their fork would change the sampler we have characterised across this work and
  would make every number incomparable with the arms it has to be read against (the
  unperturbed control at 1/20 and the s_lig gauss ablation at 9/20 on 8SGE). So TCP
  is installed on the STOCK sampler by monkeypatching, exactly as intervene.py does,
  and no file in the installed boltz package is touched.

WHERE THEIR CODE PUTS IT, AND WHERE THIS PUTS IT
  In their fork, `s_noise_fn` is handed down through network_condition_kwargs and
  applied inside DiffusionModule.forward at

      s, normed_fourier = self.single_conditioner(...)
      if s_noise_fn is not None:
          s = s_noise_fn(s)                      # bp_diffusionv2.py:141-142

  Stock Boltz-2's DiffusionModule.forward is byte-identical around that point except
  for the two hook lines, so wrapping SingleConditioning.forward and perturbing its
  own output reaches the SAME tensor at the SAME place, without rewriting any source.
  That matters for one reason beyond tidiness: `s` at this point already carries the
  Fourier timestep embedding, so TCP rotates the ligand tokens' noise-level signal
  along with everything else. Perturbing s_trunk instead (what intervene.py's s_lig arm does) does not.

THE THREE WRAPPERS, AND WHY EACH IS NEEDED
  AtomDiffusion.sample                   feats -> ligand token mask; the sigma ladder
  AtomDiffusion.preconditioned_network_forward   which step we are on -> lambda(t)
  SingleConditioning.forward             apply TCP to s

  The middle one exists because the conditioner is handed `times` but not the step
  index, and lambda is defined on the step index. Mapping sigma back to a step is
  exact -- the ladder is deterministic -- so the gate is built from the ladder in
  sample() and looked up by value rather than counted, which keeps it correct no
  matter how the sampler chunks its samples.

TWO DELIBERATE DEVIATIONS FROM THEIR CODE, BOTH RECORDED IN THE RUN LOG
  1. They draw with torch.randn_like(s), i.e. from the GLOBAL CUDA stream. That
     would shift every subsequent atom-noise draw and the arm would no longer share
     its atom-noise draws with the control arm. Here the draw comes from a private CPU
     torch.Generator, so the global stream is untouched and the comparison to the
     unperturbed arm is a comparison of the intervention alone.
  2. They draw a full-size tensor and zero the non-ligand rows; this draws the
     ligand block only. The values differ (it is noise either way); the SCALE does
     not, because `s.std()` is still taken over the whole tensor as they take it.

FROZEN MODE
  `frozen=True` draws eps ONCE per run and reuses it at every gated step. It is not
  a Boltz-Perturb option. It exists because TCP differs from our s_lig ablation in
  three ways at once -- magnitude, schedule, and that the draw is refreshed at every
  step of every sample -- and the refresh is the one that carries their diversity
  argument. Holding the draw fixed turns TCP into a single displaced conditioning
  shared by every sample, which is what the gauss ablation already is, so the pair
  separates "the conditioning moved" from "the conditioning kept moving".

  Measured on 8SGE, 20 replicates an arm: fresh 11/20 in the correct basin and 10
  poses under 2 A; frozen 7/20 and 3 under 2 A, with 4 structures below lDDT-PLI
  0.5. Same magnitude, same schedule; the redraw is the whole difference.
"""
from __future__ import annotations

import math

import torch

# Reference implementation: boltz-perturb/src/boltz/model/modules/diffusionv2.py,
# make_s_noise_fn (lines 308-347) and the lambda ramp (lines 535-538).
BP_SOURCE = ("https://github.com/MSDLLCpapers/boltz-perturb/blob/main/src/boltz/"
             "model/modules/diffusionv2.py")


def lam_at(step_idx: int, num_steps: int, noise_scale: float,
           t_start: float, t_end: float) -> float:
    """Boltz-Perturb's TADS ramp, their lines 535-538.

    steering_t runs 1 -> 0 over the trajectory, so the window [t_end, t_start] in
    steering_t is the step window [ (1-t_start)*T, (1-t_end)*T ]. lambda is LARGEST
    at the window's early edge (steering_t = t_start) and falls to zero at its late
    edge, i.e. the perturbation fades as the pose commits, not the other way round.
    """
    steering_t = 1.0 - (step_idx / num_steps)
    if t_end <= steering_t <= t_start:
        return noise_scale * (steering_t - t_end) / (t_start - t_end)
    return 0.0


def tcp(s: torch.Tensor, lig: torch.Tensor, eps: torch.Tensor, lam: float,
        psi: float, rescale: bool = True, norm_proj: bool = True) -> torch.Tensor:
    """One TCP application. A transcription of their make_s_noise_fn.

    s    [B, N, C]   the single representation AFTER the conditioner
    lig  [N] bool    ligand token rows
    eps  [B, L, C]   unit-normal draw for the L ligand rows (see the deviations note)
    lam              their `noise_scale` for THIS step, i.e. lam_at(...)
    """
    if lam <= 0:
        return s
    sd = s.float().std()                       # over the WHOLE tensor, as they do
    s_tilde = s.clone()
    s_tilde[:, lig, :] = s[:, lig, :] + (lam * sd * eps).to(s.dtype)

    s_prime = s_tilde
    if rescale:
        mu_o = s[:, lig, :].float().mean(dim=(1, 2), keepdim=True)
        sd_o = s[:, lig, :].float().std(dim=(1, 2), keepdim=True)
        mu_t = s_tilde[:, lig, :].float().mean(dim=(1, 2), keepdim=True)
        sd_t = s_tilde[:, lig, :].float().std(dim=(1, 2), keepdim=True)
        s_prime = s.clone()
        s_prime[:, lig, :] = (
            ((s_tilde[:, lig, :].float() - mu_t) / (sd_t + 1e-6)) * sd_o + mu_o
        ).to(s.dtype)

    s_hat = s.clone()
    s_hat[:, lig, :] = psi * s_prime[:, lig, :] + (1.0 - psi) * s[:, lig, :]

    if norm_proj:
        n_o = s[:, lig, :].float().norm(dim=-1, keepdim=True)
        n_p = s_hat[:, lig, :].float().norm(dim=-1, keepdim=True)
        s_hat[:, lig, :] = ((s_hat[:, lig, :].float() / (n_p + 1e-6)) * n_o).to(s.dtype)
    return s_hat


def geometry(before: torch.Tensor, after: torch.Tensor) -> dict:
    """How far the block actually moved: the quantity that makes TCP comparable
    with the gauss ablation. Rotation angle is the per-row angle, averaged."""
    a, b = before.float().flatten(0, 1), after.float().flatten(0, 1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
    return {"rotation_deg": float(torch.rad2deg(torch.acos(cos)).mean()),
            "delta_over_norm": float((b - a).norm() / (a.norm() + 1e-12))}


def install(noise_scale: float, psi: float, t_start: float, t_end: float,
            rescale: bool, norm_proj: bool, frozen: bool, seed: int, log: list):
    """Patch the three methods. Returns a state dict the runner reports from."""
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    from boltz.model.modules.encodersv2 import SingleConditioning

    orig_sample = AtomDiffusion.sample
    orig_fwd = AtomDiffusion.preconditioned_network_forward
    orig_cond = SingleConditioning.forward

    st = {"lig": None, "lam": 0.0, "gate": {}, "calls": 0, "applied": 0,
          "frozen_eps": None, "draw": 0, "steps_gated": [], "num_steps": None}

    def patched_sample(self, atom_mask, num_sampling_steps=None, **kw):
        from boltz.data import const
        n = num_sampling_steps or self.num_sampling_steps
        st["num_steps"] = n
        st["lig"] = (kw["feats"]["mol_type"]
                     == const.chain_type_ids["NONPOLYMER"])[0].bool()
        # Rebuild the sampler's own ladder so a step index can be recovered from the
        # sigma the denoiser is handed. t_hat = sigma_tm * (1 + gamma_{t}) is exactly
        # what sample() passes down, so the lookup is by value and cannot drift.
        sigmas = self.sample_schedule(n)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        st["gate"] = {}
        for i, (s_tm, g) in enumerate(zip(sigmas[:-1], gammas[1:])):
            lam = lam_at(i, n, noise_scale, t_start, t_end)
            st["gate"][round(float(s_tm) * (1.0 + float(g)), 6)] = (i, lam)
        log.append({"event": "schedule", "num_steps": n,
                    "n_ligand_tokens": int(st["lig"].sum()),
                    "n_tokens": int(st["lig"].numel()),
                    "lam_max": max(l for _, l in st["gate"].values()),
                    "window_steps": sorted(i for i, l in st["gate"].values() if l > 0)})
        return orig_sample(self, atom_mask, num_sampling_steps=num_sampling_steps, **kw)

    def patched_fwd(self, noised_atom_coords, sigma, network_condition_kwargs):
        value = float(sigma) if not torch.is_tensor(sigma) else float(sigma.flatten()[0])
        hit = st["gate"].get(round(value, 6))
        if hit is None and st["gate"]:
            # tolerate a float32/64 round-trip; the ladder is strictly decreasing
            k = min(st["gate"], key=lambda x: abs(x - value))
            hit = st["gate"][k] if abs(k - value) < 1e-3 * max(1.0, value) else None
        st["lam"] = 0.0 if hit is None else hit[1]
        st["step"] = None if hit is None else hit[0]
        st["calls"] += 1
        try:
            return orig_fwd(self, noised_atom_coords, sigma, network_condition_kwargs)
        finally:
            # The conditioner is reached only from inside this call. Clearing lambda
            # on the way out means any OTHER caller of SingleConditioning (the
            # confidence head, a future recycle) sees an untouched s instead of
            # whatever lambda the last denoiser step happened to leave behind.
            st["lam"] = 0.0

    def patched_cond(self, times, s_trunk, s_inputs):
        s, normed_fourier = orig_cond(self, times, s_trunk, s_inputs)
        lam, lig = st["lam"], st["lig"]
        if lam <= 0 or lig is None:
            return s, normed_fourier
        shape = (s.shape[0], int(lig.sum()), s.shape[2])
        if frozen:
            if st["frozen_eps"] is None or st["frozen_eps"].shape != shape:
                g = torch.Generator(device="cpu").manual_seed(seed)
                st["frozen_eps"] = torch.randn(shape, generator=g)
            eps = st["frozen_eps"].to(s.device)
        else:
            g = torch.Generator(device="cpu").manual_seed(seed + st["draw"])
            st["draw"] += 1
            eps = torch.randn(shape, generator=g).to(s.device)
        out = tcp(s, lig, eps, lam, psi, rescale, norm_proj)
        st["applied"] += 1
        if st["applied"] <= 3 or st["applied"] % 400 == 0:
            log.append({"event": "apply", "step": st["step"], "lam": round(lam, 4),
                        "applied_so_far": st["applied"],
                        **geometry(s[:, lig, :], out[:, lig, :])})
        return out, normed_fourier

    AtomDiffusion.sample = patched_sample
    AtomDiffusion.preconditioned_network_forward = patched_fwd
    SingleConditioning.forward = patched_cond
    return st
