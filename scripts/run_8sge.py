#!/usr/bin/env python3
"""[reproduce] Run the 8SGE control and gauss-s_lig arms through Boltz-2.

Needs a GPU, a Boltz-2 install, and the Boltz-2 checkpoint and cache, which are
machine-local and are named by --checkpoint / --cache or by $BOLTZ_CHECKPOINT and
$BOLTZ_CACHE. Nothing here is committed: every run writes under runs/, which is
gitignored.

The two arms:

  control  the model as shipped. No region is perturbed.
  gauss    the ligand rows of the trunk single representation are replaced by
           per-channel Gaussian noise matched to that run's own s_lig moments,
           at the trunk-to-diffusion boundary.

Each draw differs only in its direction seed, which fixes the noise. The model
seed is held fixed across an arm so that the conditioning is identical and the
draw is the only thing that varies.

    python scripts/run_8sge.py --arm control --draws 30
    python scripts/run_8sge.py --arm gauss   --draws 40

Runs are resumable: a draw whose output directory already holds collection.json
is skipped.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference" / "8sge"
RUNS = ROOT / "runs"


def prepare_inputs() -> Path:
    """Materialise the Boltz input directory, unpacking the vendored MSA.

    The MSA is stored gzipped and the YAML is stored as a template, because a
    committed YAML would have to carry an absolute path to the MSA and would
    therefore only work on the machine that wrote it.
    """

    # Named "inputs" because Boltz derives its output directory from this
    # name: boltz_results_inputs, which is what score_8sge.py looks for.
    out = RUNS / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    msa = REFERENCE / "msa_1A.csv"
    if not msa.is_file():
        with gzip.open(REFERENCE / "msa_1A.csv.gz", "rb") as src, msa.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    template = (REFERENCE / "input.yaml.template").read_text()
    (out / "8sge__1__1.A__1.H__1.H.yaml").write_text(
        template.replace("{MSA_PATH}", str(msa.resolve()))
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("control", "gauss"), required=True)
    ap.add_argument("--draws", type=int, default=40,
                    help="number of direction seeds, 1..N")
    ap.add_argument("--seed", type=int, default=42,
                    help="model seed, held fixed across the arm")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out-root", default=str(RUNS))
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter with boltz installed; defaults to this one")
    args = ap.parse_args()

    inputs = prepare_inputs()
    out_root = Path(args.out_root)
    passthrough: list[str] = []
    for flag, value in (("--checkpoint", args.checkpoint), ("--cache", args.cache)):
        if value:
            passthrough += [flag, value]

    failures = 0
    for draw in range(1, args.draws + 1):
        name = f"{args.arm}-s{args.seed:02d}_d{draw:02d}"
        out_dir = out_root / name
        if (out_dir / "collection.json").is_file():
            print(f"[{draw}/{args.draws}] {name} already done, skipping")
            continue
        # Invoked by file path, not with -m: the Boltz environment is a separate
        # interpreter that does not have this package installed, and the runner
        # has no first-party imports, so it runs standalone.
        cmd = [args.python, str(ROOT / "src" / "cofolding_tts" / "intervene.py"),
               "--inputs", str(inputs),
               "--seed", str(args.seed),
               "--lam", "0",
               "--experiment-id", "8sge_slig_rescue",
               "--out-dir", str(out_dir),
               "--log", str(out_root / "logs" / f"{name}.json")]
        if args.arm == "gauss":
            cmd += ["--regions", "s_lig", "--intervention", "gauss",
                    "--direction-seed", str(draw)]
        else:
            cmd += ["--regions", "", "--intervention", "additive"]
        cmd += passthrough
        (out_root / "logs").mkdir(parents=True, exist_ok=True)
        print(f"[{draw}/{args.draws}] {name}")
        if subprocess.run(cmd).returncode != 0:
            failures += 1
            print(f"[{draw}/{args.draws}] {name} FAILED", file=sys.stderr)
    print(f"{args.arm} arm complete, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
