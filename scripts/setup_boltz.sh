#!/usr/bin/env bash
# Install Boltz-2 for the reproduce tier, in its OWN environment.
#
# Why separate: Boltz pins a CUDA-specific torch build. Installing it beside the
# inspect tier's dependencies is the fastest way to end up with a torch that does not
# match the GPU, and the inspect tier is meant to keep working with no torch at all.
#
# This DOWNLOADS several hundred MB of wheels, and the first prediction then downloads
# the Boltz cache (several GB) and the checkpoint. Nothing is vendored in this
# repository.
#
#   bash scripts/setup_boltz.sh [venv-path]     # default: .venv-boltz
set -euo pipefail

VERSION="2.2.1"                      # every number in this repository was produced here
VENV="${1:-.venv-boltz}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -d "$VENV" ]; then
  echo "$VENV already exists; delete it or pass another path." >&2
  exit 1
fi

echo "Creating $VENV and installing boltz==$VERSION (this downloads a lot)..."
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install "boltz==$VERSION"

echo
"$VENV/bin/python" - <<'PY'
import importlib, importlib.metadata as m
print("boltz", m.version("boltz"))
for mod, cls in (("boltz.model.modules.diffusionv2", "AtomDiffusion"),
                 ("boltz.model.modules.encodersv2", "SingleConditioning"),
                 ("boltz.model.modules.diffusion_conditioning", "DiffusionConditioning")):
    getattr(importlib.import_module(mod), cls)
    print("  patch target present:", f"{mod}.{cls}")
PY

cat <<EOF

Done. The reproduce tier runs with that interpreter:

  export BOLTZ_CHECKPOINT=/path/to/boltz2_conf.ckpt
  export BOLTZ_CACHE=/path/to/boltz/cache
  python scripts/run_8sge.py --arm control --draws 41 --python $VENV/bin/python

The checkpoint and cache are not vendored. Boltz downloads its own cache on the first
prediction; point BOLTZ_CACHE somewhere with room for it.
EOF
