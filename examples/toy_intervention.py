#!/usr/bin/env python3
"""A 60-line analogy of what the s_lig intervention does, with numpy only.

This is NOT a simulation of Boltz-2. It is a toy with the same shape, so that the
mechanism is visible without a GPU:

    ToyFolder.trunk   stands in for the trunk: it produces one row per token
    ToyFolder.sample  stands in for the diffusion module: it reads those rows
                      and decides where the ligand goes

The toy's "decision" is deliberately made from the DIFFERENCE between two ligand
rows, not from their average. That is the part of the real finding worth feeling:
the commitment lives in how the rows differ from each other, and the per-channel
mean carries none of it. So an intervention that preserves the channel statistics
and destroys the token-to-token structure erases the commitment completely.

    python examples/toy_intervention.py
"""

from __future__ import annotations

import numpy as np

N_PROTEIN, N_LIGAND, N_CHANNEL = 8, 4, 6


class ToyFolder:
    """Stands in for the model. Nothing here is ever edited."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def trunk(self, seed: int) -> np.ndarray:
        """One row per token. The ligand rows encode a committed, wrong answer."""
        rng = np.random.default_rng(seed)
        s = rng.normal(0.0, 1.0, (N_PROTEIN + N_LIGAND, N_CHANNEL))
        # Bake the commitment into the DIFFERENCE between the first and last
        # ligand row, leaving the per-channel mean of the block untouched.
        bias = 3.0 * self.weights
        s[N_PROTEIN] -= bias
        s[-1] += bias
        return s

    def sample(self, s: np.ndarray) -> int:
        """Return +1 for the correct placement, -1 for the wrong one."""
        lig = s[N_PROTEIN:]
        return int(np.sign((lig[0] - lig[-1]) @ self.weights))


def ligand_span(s: np.ndarray) -> tuple[int, int]:
    return N_PROTEIN, len(s)


def install(model: ToyFolder, intervention: str, log: list) -> None:
    """Replace the NAME `sample` on the class. The original stays intact."""

    original = ToyFolder.sample                     # keep a reference to the real one

    def patched(self, s, *, draw: int = 0):
        if intervention == "gauss":
            lo, hi = ligand_span(s)
            s = s.copy()
            block = s[lo:hi]
            mu = block.mean(axis=0)                 # per-channel mean, kept
            sd = block.std(axis=0)                  # per-channel spread, kept
            eps = np.random.default_rng(draw).normal(0.0, 1.0, block.shape)
            s[lo:hi] = mu + sd * eps                # token structure, destroyed
            log.append({"draw": draw, "shape": block.shape,
                        "base_norm": float(np.linalg.norm(block)),
                        "new_norm": float(np.linalg.norm(s[lo:hi]))})
        return original(self, s)                    # hand off to the real method

    ToyFolder.sample = patched                      # the whole "rewrite"


def main() -> None:
    rng = np.random.default_rng(0)
    model = ToyFolder(rng.normal(0.0, 1.0, N_CHANNEL))
    pristine = ToyFolder.sample                     # for the identity check below

    print("CONTROL  (no intervention installed)")
    outcomes = [model.sample(model.trunk(seed)) for seed in range(20)]
    print(f"  20 conditionings -> correct in {outcomes.count(1)} of 20")
    print(f"  outcomes: {''.join('+' if o > 0 else '-' for o in outcomes)}")
    print("  the model is committed: the conditioning never changes the answer\n")

    log: list = []
    install(model, "gauss", log)
    print("GAUSS  (ligand rows replaced, one fixed conditioning, 20 draws)")
    s = model.trunk(seed=0)
    outcomes = [model.sample(s, draw=d) for d in range(20)]
    print(f"  20 draws -> correct in {outcomes.count(1)} of 20")
    print(f"  outcomes: {''.join('+' if o > 0 else '-' for o in outcomes)}")
    print("  the commitment is gone; the draw decides\n")

    block = s[N_PROTEIN:]
    print("WHAT THE INTERVENTION KEPT AND WHAT IT DESTROYED")
    rebuilt = block.mean(0) + block.std(0) * np.random.default_rng(0).normal(
        size=block.shape)
    print(f"  per-channel mean before : {np.round(block.mean(0), 2)}")
    print(f"  per-channel mean after  : {np.round(rebuilt.mean(0), 2)}  (kept, up to sampling)")
    print(f"  row 0 minus row -1 before: {np.round(block[0] - block[-1], 2)}")
    print(f"  row 0 minus row -1 after : {np.round(rebuilt[0] - rebuilt[-1], 2)}  (destroyed)")
    print(f"  block norm {log[0]['base_norm']:.1f} -> {log[0]['new_norm']:.1f}\n")

    print("THE ORIGINAL METHOD WAS NEVER MODIFIED")
    print(f"  saved reference is boltz-equivalent original : {pristine.__name__}")
    print(f"  class attribute now points at               : {ToyFolder.sample.__name__}")
    print(f"  they are different objects                  : {pristine is not ToyFolder.sample}")
    ToyFolder.sample = pristine
    print(f"  restored, control answer again              : {model.sample(model.trunk(0))}")


if __name__ == "__main__":
    main()
