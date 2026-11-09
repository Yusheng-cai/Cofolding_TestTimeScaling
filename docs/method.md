# What the intervention is

## Where it is installed

Boltz-2 runs a trunk and then a diffusion module. The trunk produces a single
representation `s` with one row per token, and the diffusion module is conditioned
on it. `s_lig` is the slice of those rows belonging to the ligand: for 8SGE that is
33 rows of 384 channels, one row per heavy atom.

The intervention replaces that slice, once, at the moment the trunk hands off to
diffusion. Nothing before it changes and nothing after it is touched. The protein
rows are left exactly as the trunk produced them, which is why the fold is
unaffected and only the ligand moves.

## The four interventions

Each answers a different question, which is why more than one exists.

| name | what it does | what it tests |
|---|---|---|
| `gauss` | every row becomes an independent Gaussian sample matched to the slice's own per-channel mean and standard deviation | destroys which token owns which row, keeps the channel statistics |
| `mean` | every row becomes the slice's per-channel mean row | the same destruction, deterministically, with no noise drawn |
| `shuffle` | the rows are permuted among themselves | destroys ownership while every row stays one the trunk actually produced |
| `additive` | adds `lambda * sigma * eps` | the original signal survives, buried under noise; the only one for which `--lam` means anything |

`gauss` is the one the rescue uses. A result that holds under both `gauss` and
`mean` cannot be an artifact of the particular noise drawn, because `mean` draws
none.

## What `s_lig` turns out to contain

Decompose a row as `s_lig[t, c] = mu[c] + r[t, c]`: a per-channel mean that every
token shares, plus a token-varying residual. `gauss` preserves `mu` and the
per-channel spread and destroys `r`.

On 8SGE this is enough to free the axial choice. The unperturbed model commits to
one sense of the hindered axis and essentially never leaves it; with `r` destroyed,
the direction seed decides, and about half the draws land on the crystal's side.
So the commitment lives in the token-varying part, not in the channel statistics.

## Seeds

Two seeds matter and they are independent.

The **model seed** fixes the conditioning: the MSA subsample and everything the
trunk does. Holding it fixed across an arm is what makes the draws comparable.

The **direction seed** fixes the noise `eps` that replaces the slice. It is derived
from the seed and the region name alone, so the same direction seed gives the same
`eps` regardless of the model seed. That is what lets two arms at different model
seeds be compared draw by draw.

## What this is not

It is not a method yet. Destroying a representation helps on a system the model
gets wrong; whether it harms systems the model already gets right is a separate
question, and a negative answer there would make this vandalism that happened to
pay out. That control belongs in this repository as soon as it is run.

There is also a published collision worth knowing about: Boltz-Perturb reports the
same high-level mechanism by time-annealed noise injection rather than replacement,
scored against an oracle. A reference-free endpoint and a region-resolved
intervention are what distinguish this line, and both need to hold on more than one
system.
