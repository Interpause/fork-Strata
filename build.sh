#!/usr/bin/env bash
# Rebuild every compiled part of Strata from scratch, then stop (nothing is started, no model is
# touched): setup.py --build compiles instead of using the ready-made engine, and --no-start ends the
# run after the build. setup.sh --build and update.sh skip the compile while the engine's build stamp
# still matches the source; this deletes the build folders and the stamps first, so the compile always
# happens. The Python environment, ccache and the model files are left as they are.
#
#   ./build.sh                     # the GPU setup detects, engine only
#   ./build.sh --vision gpu        # anything else goes to setup.py unchanged
#   ./build.sh --yes --gpu 0       # (--build and --no-start are always added)
set -uo pipefail
cd "$(dirname "$0")" || exit 1

for a in "$@"; do
  case "$a" in
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  esac
done

echo "Rebuilding Strata from scratch."
# The engine's CMake folders (CUDA 13, the experimental CUDA 12, HIP, and the image encoders) and
# the stamps setup.py checks before it decides the engine is up to date: with a stamp gone the
# engine is compiled again even though the source hash has not changed.
for d in build build-cuda12 build-hip build-vision build-vision-cuda12; do
  if [[ -e $d ]]; then echo "  removing $d/"; rm -rf "$d"; fi
done
for f in engine/BUILD.json engine-cuda12/BUILD.json; do
  if [[ -e $f ]]; then echo "  removing $f"; rm -f "$f"; fi
done
echo

# --build: compile instead of taking a ready-made engine. --no-start: end there, before the model
# (or the benchmark offer) is started. setup.sh creates .venv when it is missing and hands the rest to
# setup.py.
exec sh ./setup.sh --build --no-start "$@"
