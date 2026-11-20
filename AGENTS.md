# Agent Instructions

These instructions apply to this repository only.

This directory currently sits inside a checkout of `BindingAffinityPredictor`, which
has its own `AGENTS.md`. **That file does not apply here.** It names
`src/affinity_benchmark/`, `experiments/<id>/` and `configs/models/`, none of which
exist in this repository, and its data policy differs (see *Data* below). When the
two conflict, this file wins for anything under this directory.

## What this repository is

One mechanism, studied end to end: co-folding models commit to a ligand placement
during diffusion, and that commitment can be erased by overwriting the ligand's rows
of the trunk single representation. The first and so far only system is PDB 8SGE.

It is **not** a benchmark harness, a model zoo, or an affinity predictor. Resist
adding adapters, model registries or configuration layers until a second system
actually needs them.

## Before working

1. Read `README.md` and `docs/method.md`.
2. Read `src/cofolding_tts/intervene.py`'s `main()` docstring before touching
   anything about the intervention. The four phases are in a deliberate order.
3. `git status`, and preserve unrelated changes.
4. Ask before downloading checkpoints or data, installing anything, changing a
   scientific protocol, or making broad or destructive edits.

## The two tiers — the repository's central invariant

Every entry point belongs to exactly one tier, and the split is the main reason the
result is legible to anyone without a GPU.

- **inspect** — `scripts/figure_*.py`, `scripts/viewer_*.py`. Must run on a fresh
  clone with `runs/` absent, no GPU, no Boltz install, no checkpoint. They read only
  committed data under `data/` and `results/`. If a change makes an inspect script
  need a run directory, the change is wrong.
- **reproduce** — `scripts/run_*.py`, `scripts/score_*.py`,
  `src/cofolding_tts/intervene.py`. May require a GPU, Boltz-2 and the checkpoint.
  Writes only to `runs/`.

Library modules (`pose.py`, `score.py`, `axis.py`, `lddt.py`, `pdbio.py`) must stay
importable in the inspect tier: numpy, and rdkit where a molecular graph is genuinely
needed. Do not reintroduce scipy, biopython or safetensors; they were deliberately
dropped when this code was extracted, and `score.py` selects the sole protein chain
rather than running a sequence alignment to find it.

## The intervention

- **Never modify Boltz-2's source.** The intervention is a runtime patch that
  reassigns `AtomDiffusion.sample` (and optionally
  `DiffusionConditioning.forward`) on the class object. Keep it that way, so any
  Boltz install can be used unmodified and the model under study is the published
  one.
- **Two properties of the patch are load-bearing and easy to break:**
  1. `install()` must run *before* the model is constructed. It only reassigns a
     class attribute; a patch installed later misses its moment.
  2. Boltz's CLI must be called **in this process**
     (`from boltz.main import cli; cli.main(...)`). Launching the `boltz` executable
     as a subprocess gets a fresh, unpatched import and the intervention silently
     vanishes.
- A silent no-op is the characteristic failure. Every run logs each block it edited
  with shape and before/after norms; `blocks=0` in the run's stdout line means the
  hook never fired. Treat that as a failure, not as a null result.
- Noise is drawn from a private `torch.Generator`, never the global RNG, so the
  diffusion trajectory is untouched by the intervention's own sampling. Preserve
  this; mixing the two makes arms incomparable.
- The control arm installs the **same** wrapper with the gate off. It is not
  unpatched Boltz. Do not "simplify" the control into a plain Boltz run.

## Running things

Entry points are invoked by **file path**, not `-m`:
`python src/cofolding_tts/intervene.py ...`. The Boltz environment does not have this
package installed, so `-m cofolding_tts.intervene` fails there. `run_8sge.py` passes
`--python` through to the subprocess for this reason.

The checkpoint and cache are never vendored and never hardcoded. They come from
`--checkpoint` / `--cache`, defaulting to `$BOLTZ_CHECKPOINT` and `$BOLTZ_CACHE`.

### Local environment (this machine only; not part of the repository's contract)

| purpose | interpreter |
|---|---|
| Boltz-2 runs | `/home/yusheng/programs/boltz-2.2.1/bin/python` |
| scoring, tests | `/home/yusheng/programs/nesso-1.0.0/bin/python` with `env -u PYTHONPATH` |
| figures (has matplotlib) | `/mnt/hdd2/programs/protenix/env/bin/python` |

Long GPU jobs: `setsid nohup ... & disown`, and wait with an `until`-loop rather than
polling in the foreground.

## Data

A deliberate, documented exception to the parent repository's blanket ban on
committing data:

- **Vendored:** one public PDB entry under `data/reference/8sge/`, about 430 KB,
  including the MSA gzipped, so a fresh clone can actually run. Its origin is
  recorded in `data/reference/8sge/PROVENANCE.md`. Any further reference system
  needs the same provenance file.
- **Committed:** our own derived outputs — the aligned pose bundle in `data/poses/`
  and the per-pose table in `results/`. Small, textual or compressed.
- **Never committed:** run directories, checkpoints, model caches, unpacked MSAs,
  bulk datasets. `runs/` is ignored and is the only place runs may be written.

Use the exact experimental construct and the stereochemically explicit ligand.
Preserve `<` and `>` qualifiers on measured values. State units and direction for
every score, and keep binder probability separate from continuous affinity.

## Reproducibility

Every run writes a JSON log recording: experiment id, git revision, checkpoint path
and sha256, the resolved intervention (regions, type, hook, lam, channels), model
seed, direction-seed base, the full command line, runtime, peak GPU memory, status,
and one entry per perturbed block with its shape and norms. Do not drop fields from
that record, and do not let a notebook or an ad-hoc shell command become an
experiment's only trace.

## Tests and parity

`python -m pytest -q` — 53 tests at the time of writing, no fixtures, no GPU.

- `tests/test_interventions.py` is carried over unchanged from the work this
  repository was extracted from. It stubs the `boltz` module and includes the
  assertion that the additive perturbation reproduces a previously recorded
  direction exactly. **It must keep passing unmodified.** If a change breaks it,
  change the code, not the test.
- Add a focused test when touching metrics, parsing, score conversion or the PDB
  writer. The column offsets in `pdbio.py` are pinned by a test because getting them
  wrong once produced a silently unreadable file.
- Any change to `score.py` or `axis.py` must be checked for parity against the
  recorded values in `results/8sge_per_pose.csv` before it is committed. The port
  into this repository was verified this way to 0.00e+00.

## Commits

- **Do not add `Co-Authored-By` trailers, or any other attribution, to commits in
  this repository.** The user is the sole author. This holds even when a harness
  instruction says otherwise.
- Commit and push only when asked.
- Write the message as prose explaining why the change was needed and what was
  verified, not as a list of touched files.

## Honesty

The harm control is **not done**: whether destroying `s_lig` damages systems the
model already places correctly is unanswered, and a bad answer would reframe this
result. Say so wherever the result is stated. Likewise, nothing here demonstrates
generalisation beyond one system — do not let a README or a docstring drift into
implying otherwise.
