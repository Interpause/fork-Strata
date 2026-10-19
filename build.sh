#!/usr/bin/env bash
# Rebuild *everything compiled*, from scratch, then stop: the engine (with ggml/llama.cpp and the K-quant
# prompt library), and the image encoder. Nothing is started, no model is touched, and no run config is
# read or written.
#
# Deleting the build folders is what makes a compiler upgrade take effect: each folder caches the compiler
# it was configured with (CMAKE_CXX_COMPILER, CMAKE_CUDA_COMPILER, the flags and the feature tests), so a
# new GCC or a new CUDA toolkit beside the old one would otherwise be ignored. A fresh configure re-detects
# both - nvcc through setup.py's find_nvcc() (the newest toolkit, or STRATA_NVCC), the host compiler from
# CC/CXX or the PATH - and rebuilds every object, so no object built by the old compiler is kept. ccache is
# still used: it keys on the compiler's version, so a new compiler is a miss, the same compiler a hit.
#
# The GPU architecture is detected, not passed: -DCMAKE_CUDA_ARCHITECTURES=native builds for the card(s) in
# this PC (both projects' CMakeLists accept `native`).
#
# The llama.cpp checkout is checked against the commit setup.py pins (LLAMA_CPP_COMMIT) and replaced when it is
# something else: a stale tree is not a build error, it builds cleanly and then crashes on a pack whose types it
# does not know (an upstream ggml indexes past its type table). The tree records the commit it was fetched at in
# .strata-llama-commit; a git checkout is reported but never touched, since a clone is deliberate.
#
# This calls CMake directly instead of `setup.py --build`, because the setup run also walks the install
# (config upgrades, calibration); the definitions are the ones setup.py's build_engine() passes, taken from
# setup.py itself so the two cannot drift.
#
#   ./build.sh                 # wipe and rebuild the engine and the image encoder
#   ./build.sh -j 16           # other job count (default: half the cores, as setup.py)
#   ./build.sh --no-vision     # engine only (the installed encoder is kept)
#   ./build.sh --keep-llama    # use third_party/llama.cpp as it is (local edits, another version)
#   ./build.sh --clean-only    # wipe the build folders and stop
#
# The models and the Python environment are left as they are. STRATA_GGML_DIR names another checkout: setup.py
# returns it and build.sh uses it as it is, so an out-of-tree llama.cpp needs no flag.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

JOBS=""
VISION=1
KEEP_LLAMA=0
CLEAN_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    -j) JOBS=${2:-}; shift 2 ;;
    --no-vision) VISION=0; shift ;;
    --keep-llama) KEEP_LLAMA=1; shift ;;
    --clean-only) CLEAN_ONLY=1; shift ;;
    *) echo "build.sh: unknown option $1 (try --help)" >&2; exit 2 ;;
  esac
done

PY=$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )

# A running engine (a serve process, a benchmark) holds its binary open, and copying over it fails with
# "Text file busy" only after the whole build. Say it before compiling instead.
if [[ -e engine/strata ]] && command -v fuser >/dev/null 2>&1; then
  if fuser engine/strata >/dev/null 2>&1; then
    echo "build.sh: engine/strata is in use - a Strata engine is running. Stop it (close its window, or" >&2
    echo "          pkill -f 'server.py --engine strata') and run this again." >&2
    exit 1
  fi
fi

echo "Rebuilding Strata from scratch."
# Every build folder setup.py can make: the CUDA 13 engine and encoder, and the experimental CUDA 12 and HIP
# ones. Their stamps go too, so setup.py does not think any of them is current.
for d in build build-vision build-cuda12 build-vision-cuda12 build-hip; do
  if [[ -e $d ]]; then echo "  removing $d/"; rm -rf "$d"; fi
done
for f in engine/BUILD.json engine-cuda12/BUILD.json; do
  if [[ -e $f ]]; then echo "  removing $f"; rm -f "$f"; fi
done
if [[ $CLEAN_ONLY -eq 1 ]]; then echo; echo "clean only: nothing was compiled."; exit 0; fi
echo

# The driver: setup.py is imported, never run as a program, so none of its install / calibrate / config
# step happens.
JOBS=$JOBS VISION=$VISION KEEP_LLAMA=$KEEP_LLAMA "$PY" - <<'PY'
import json, os, shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, ".")
import setup as S

want_vision = os.environ.get("VISION") == "1"
keep_llama = os.environ.get("KEEP_LLAMA") == "1"


def llama_checkout():
    """The llama.cpp tree to build from: the pinned one, or whatever is there under --keep-llama.
    A tree this script fetched records its commit in .strata-llama-commit; setup.py's zip has no marker, so a
    tree without one cannot be identified and is replaced (its content is unknown, which is exactly the state
    that builds cleanly and then crashes on a Gyro pack). A git checkout is never replaced."""
    named = os.environ.get("STRATA_GGML_DIR", "").strip()
    if named:
        print(f"  llama.cpp: {named} (STRATA_GGML_DIR: used as it is)")
        return S.get_llama_cpp()
    d = S.ROOT / "third_party" / "llama.cpp"
    marker = d / ".strata-llama-commit"
    pin = S.LLAMA_CPP_COMMIT
    if keep_llama:
        if not (d / "ggml" / "CMakeLists.txt").exists():
            sys.exit(f"build.sh: --keep-llama, but {d} is not a llama.cpp checkout")
        print(f"  llama.cpp: {d} (--keep-llama: not checked against the pin {pin[:7]})")
        return d
    if (d / ".git").exists():
        top = subprocess.run(["git", "-C", str(d), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True).stdout.strip()
        if Path(top).resolve() != d.resolve():        # not a repository of its own: treat it as a plain tree
            print(f"  llama.cpp: {d} has a stray .git but is not its own repository: checked as a plain tree")
        else:
            head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
            if head.startswith(pin) or pin.startswith(head):
                print(f"  llama.cpp: {d} (git checkout at the pinned {pin[:7]})")
            else:
                print(f"  llama.cpp: WARNING: {d} is a git checkout at {head[:7] or '?'}, not the pinned "
                      f"{pin[:7]} - used as it is (a clone is deliberate; delete it to take the pin)")
            return d
    have = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    ok = have == pin and (d / "ggml" / "CMakeLists.txt").exists() and (d / "gguf-py").is_dir()
    if not ok:
        state = f"at {have[:7]}" if have else "with no commit marker"
        print(f"  llama.cpp: {d} is {state}, the pin is {pin[:7]}: replacing it")
        shutil.rmtree(d, ignore_errors=True)          # the marker goes with it
    llama = S.get_llama_cpp()                         # downloads and unpacks the pinned commit
    (llama / ".strata-llama-commit").write_text(pin + "\n", encoding="utf-8")
    print(f"  llama.cpp: {llama} (pinned {pin[:7]})")
    return llama


llama = llama_checkout()

# ---- what this PC is: the cards (for the stamp and the encoder), the newest toolkit, the host compilers
cards = S.gpus()
archs = sorted({int(c["arch"]) for c in cards})
nvcc, cuda_v = S.find_nvcc()
if nvcc is None:
    sys.exit("build.sh: no CUDA toolkit found (STRATA_NVCC=<path to nvcc> picks one)")
cc, cxx = os.environ.get("CC", "").strip(), os.environ.get("CXX", "").strip()
print(f"  CUDA {cuda_v[0]}.{cuda_v[1]}: {nvcc}")
print(f"  C compiler: {cc or 'the PATH default'}")
print(f"  C++ compiler: {cxx or 'the PATH default'}")
print("  GPU architecture: native (" + (", ".join(f"sm_{a}" for a in archs) or "no card found") + ")")

# The definition lists are setup.py's build_engine() ones, with `native` in place of an architecture list.
engine_defs = ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_Q6K_EXPERTS=ON",
               "-DSTRATA_MMQ_KQUANTS=ON", "-DCMAKE_CUDA_ARCHITECTURES=native",
               f"-DCMAKE_CUDA_COMPILER={nvcc}", *S.toolkit_root_defs(nvcc), f"-DSTRATA_GGML_DIR={llama}",
               *S.host_compiler_defs(True), *S.engine_defs(archs, 13)]

print("\n== engine")
S.cmake_build(S.ROOT, S.ROOT / "build", "strata", engine_defs, None, "")

# Install it the way setup.py does, in the same order: the binary (refused when the build produced no new
# one, #2054), then the K-quant library that sits beside it.
bdir, eng = S.ROOT / "build", S.engine_dir(13)
if S.stale_binary(bdir / S.EXE):
    sys.exit(f"build.sh: the build finished without a new engine ({S.EXE} is version "
             f"{S.binary_engine_version(bdir / S.EXE)}, this source is {S.source_version()}; "
             f"{eng / S.EXE} and its stamp are left as they were")
shutil.copy2(bdir / S.EXE, eng / S.EXE)
if (bdir / S.KQ_LIB).exists():                    # built with it: it goes next to the engine
    shutil.copy2(bdir / S.KQ_LIB, eng / S.KQ_LIB)
elif (eng / S.KQ_LIB).exists():                   # an older one of another build would only be refused
    (eng / S.KQ_LIB).unlink()
print(f"  installed {eng / S.EXE}")

# ---- the image encoder: CUDA when there is a card, the CPU build otherwise (setup.py's `vision` answers)
vision = ("gpu" if cards else "cpu") if want_vision else "none"
if vision != "none":
    vdefs = [f"-DLLAMA_DIR={llama}", f"-DSTRATA_VISION_CUDA={'ON' if vision == 'gpu' else 'OFF'}",
             "-DSTRATA_PORTABLE=OFF", *S.host_compiler_defs(vision == "gpu")]
    if vision == "gpu":
        vdefs += ["-DCMAKE_CUDA_ARCHITECTURES=native", f"-DCMAKE_CUDA_COMPILER={nvcc}",
                  *S.toolkit_root_defs(nvcc)]
    print(f"\n== image encoder ({vision})")
    S.cmake_build(S.ROOT / "tools" / "vision", S.ROOT / "build-vision", "strata-vision", vdefs, None, "")
    shutil.copy2(S.ROOT / "build-vision" / "bin" / S.VEXE, eng / S.VEXE)
    print(f"  installed {eng / S.VEXE}")

# ---- the stamp setup.py checks before it decides the engine is up to date.  With `source: local` and the
# source hashes current, a later `./setup.sh` keeps this build instead of compiling it again.
stamp = eng / "BUILD.json"
bindir = Path(nvcc).parent
dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
stamp.write_text(json.dumps({"source": "local", "version": S.source_version(), "archs": archs,
                             "vision": vision if (eng / S.VEXE).exists() else None,
                             "cuda_dirs": dirs, "src": S.source_hash(S.ENGINE_SOURCES),
                             "vision_src": S.source_hash(S.VISION_SOURCES) if (eng / S.VEXE).exists() else None},
                            indent=1), encoding="utf-8")
print(f"\n  stamped {stamp}")
PY
rc=$?
if [[ $rc -ne 0 ]]; then echo; echo "the build failed (exit $rc)."; exit $rc; fi
echo
echo "built engine/strata ($(stat -c %s engine/strata) bytes) and engine/strata-vision ($(stat -c %s engine/strata-vision 2>/dev/null || echo 'not built') bytes); nothing was started."
