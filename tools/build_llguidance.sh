#!/usr/bin/env bash
# Build the pinned llguidance library the grammar needs.
#
#     tools/build_llguidance.sh
#
# llguidance is a Rust crate with a C API.  Strata pins one version of it and builds it from source, the way
# setup.py builds llama.cpp: the checkout lives under third_party/ (gitignored) and is reused on every later run.
# The first run clones the pin and vendors every crate it needs, so all later runs build offline.
#
# STRATA_LLGUIDANCE_DIR names the checkout to use instead of third_party/llguidance, the way STRATA_GGML_DIR names
# a llama.cpp checkout.  The script is idempotent: a second run finds the checkout, the vendor tree and the build
# directory already there and only rebuilds what cargo thinks is stale.
#
# The last lines print the two artefacts and the variable a consumer sets to find the shared library.
set -euo pipefail

REMOTE=https://github.com/guidance-ai/llguidance.git
PIN=f0971424ec072d3e4d4196bcc7f31a2f60527df9        # v1.9.1
PIN_NAME=v1.9.1
# The clone's own rust-toolchain.toml pins 1.95.0, which is older than the parser needs.  Asking for the toolchain
# on the command line builds with a newer one and leaves that file alone, so the checkout stays the upstream pin.
TOOLCHAIN=1.99.1

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET=${STRATA_LLGUIDANCE_DIR:-$REPO/third_party/llguidance}

say() { printf '%s\n' "$*"; }
fail() { printf 'build_llguidance: %s\n' "$*" >&2; exit 1; }

# Milliseconds, for the step timings.  GNU date has %N; a date without it prints the literal, so fall back to seconds.
now_ms() {
    local t
    t=$(date +%s%N 2>/dev/null || true)
    case "$t" in
        *N|"") echo $(( $(date +%s) * 1000 )) ;;
        *)     echo $(( t / 1000000 )) ;;
    esac
}
# Elapsed milliseconds since the argument, for the "took N ms" lines.
since() { echo $(( $(now_ms) - $1 )); }

# --- 0. the toolchain --------------------------------------------------------------------------------------------
command -v git >/dev/null 2>&1 || fail "git not found"
command -v cargo >/dev/null 2>&1 || fail "cargo not found: install rustup, then 'rustup toolchain install $TOOLCHAIN'"
RUSTUP_TOOLCHAIN=$TOOLCHAIN rustc --version >/dev/null 2>&1 \
    || fail "Rust $TOOLCHAIN is not installed: rustup toolchain install $TOOLCHAIN"

say "llguidance $PIN_NAME ($PIN)"
say "checkout: $TARGET"

# --- 1. the checkout ---------------------------------------------------------------------------------------------
t0=$(now_ms)
if [ -d "$TARGET/.git" ]; then
    say "1. checkout already present, not cloned"
else
    if [ -e "$TARGET" ] && [ -n "$(ls -A "$TARGET" 2>/dev/null || true)" ]; then
        fail "$TARGET exists and is not a git checkout; move it away (or set STRATA_LLGUIDANCE_DIR) and run again"
    fi
    say "1. cloning $REMOTE"
    mkdir -p "$(dirname "$TARGET")"
    git clone --quiet "$REMOTE" "$TARGET"
fi

have=$(git -C "$TARGET" rev-parse HEAD)
if [ "$have" = "$PIN" ]; then
    say "   at the pin"
else
    # A different commit is moved to the pin, but never over edited tracked files: those are work, and replacing
    # them silently would throw it away.  Untracked files (the vendor tree, target/) are not work and stay put.
    dirty=$(git -C "$TARGET" status --porcelain --untracked-files=no)
    if [ -n "$dirty" ]; then
        fail "$TARGET is at ${have:0:12} with uncommitted changes; commit or discard them, then run again"
    fi
    if ! git -C "$TARGET" cat-file -e "$PIN^{commit}" 2>/dev/null; then
        say "   fetching the pin"
        git -C "$TARGET" fetch --quiet origin
        git -C "$TARGET" cat-file -e "$PIN^{commit}" 2>/dev/null \
            || fail "the pin $PIN is not in $TARGET after fetching origin"
    fi
    git -C "$TARGET" checkout --quiet --detach "$PIN"
    say "   checked out $PIN (was ${have:0:12})"
fi
say "   took $(since "$t0") ms"

# --- 2. the vendored crates --------------------------------------------------------------------------------------
# Vendoring is what makes the build offline and reproducible: every crate cargo would download sits in vendor/,
# and .cargo/config.toml points crates-io at it.  Either file missing means the pair is incomplete, so redo it.
t0=$(now_ms)
if [ -d "$TARGET/vendor" ] && [ -f "$TARGET/.cargo/config.toml" ]; then
    say "2. crates already vendored ($(du -sh "$TARGET/vendor" | cut -f1)), not vendored again"
else
    say "2. vendoring the crates (the only step that uses the network)"
    # At the checkout's root, not against parser/: parser is a workspace member, so resolving it pulls the whole
    # workspace, and vendoring from parser/ alone fails with "no matching package named anyhow found ...
    # required by package llguidance_py".
    ( cd "$TARGET" && RUSTUP_TOOLCHAIN=$TOOLCHAIN cargo vendor >/dev/null )
    mkdir -p "$TARGET/.cargo"
    # Cargo's shape for a vendored tree (the same block `cargo vendor` prints on stdout).  The path is absolute so
    # the file keeps working when cargo is started from a subdirectory of the checkout.
    cat > "$TARGET/.cargo/config.toml" <<EOF
# Written by Strata's tools/build_llguidance.sh.  The crates are vendored beside the source, so the build never
# reaches crates.io.  To re-vendor, delete this file and vendor/ and run the script again.
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "$TARGET/vendor"
EOF
    say "   vendored into $TARGET/vendor"
fi
say "   took $(since "$t0") ms"

# --- 3. the build ------------------------------------------------------------------------------------------------
t0=$(now_ms)
say "3. building (release, offline, from the vendor tree)"
( cd "$TARGET" && RUSTUP_TOOLCHAIN=$TOOLCHAIN \
    cargo build --release --locked --offline --manifest-path parser/Cargo.toml )
say "   took $(since "$t0") ms"

# --- 4. the artefacts --------------------------------------------------------------------------------------------
SO=$TARGET/target/release/libllguidance.so
A=$TARGET/target/release/libllguidance.a
[ -f "$SO" ] || fail "the build produced no $SO"
[ -f "$A" ] || fail "the build produced no $A"

size() { wc -c < "$1" | tr -d ' '; }
say "4. artefacts"
say "   $SO ($(size "$SO") bytes)"
say "   $A ($(size "$A") bytes)"
say ""
say "A consumer points at the shared library with:"
say "    STRATA_LLGUIDANCE_LIB=$SO"
