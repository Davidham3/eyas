#!/usr/bin/env bash
# Local release preflight: prove the same build/install story the release
# workflow relies on, before a tag ever reaches CI.
#
# why: CI builds once and smokes the published TestPyPI artifacts, but the pip
# user's minimal story -- "install the wheel, import eyas, run the CLI" -- is
# never exercised against a locally built artifact. A checkout hides packaging
# faults (src layout, missing package data, console-script wiring); a wheel does
# not. This script closes that gap with a clean venv.
#
# what: cleans dist/, builds sdist + wheel with `uv build`, checks both with
# `twine check`, installs the exact built wheel (never `-e`) into a throwaway
# venv at /tmp/eyas-preflight-venv, then imports eyas, verifies `eyas --version`
# against the installed distribution metadata, and runs `eyas --help` from that
# venv. Every step prints a marker; any failure exits non-zero.
#
# Usage (any directory; repo root is resolved from this file's location):
#     bash scripts/release_preflight.sh
#
# Idempotent: a stale dist/ is removed and an existing venv is recreated, so
# back-to-back runs behave identically.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$REPO_ROOT/dist"
VENV="/tmp/eyas-preflight-venv"

cd "$REPO_ROOT"

step() {
  printf '\n==> %s\n' "$*"
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

# --- 1/5: build ------------------------------------------------------------
step "1/5 Clean dist/ and build sdist + wheel (uv build)"
rm -rf "$DIST"
uv build

shopt -s nullglob
wheels=("$DIST"/eyas-*.whl)
sdists=("$DIST"/eyas-*.tar.gz)
shopt -u nullglob

[[ ${#wheels[@]} -eq 1 ]] || fail "expected exactly one wheel in dist/, found ${#wheels[@]}"
[[ ${#sdists[@]} -eq 1 ]] || fail "expected exactly one sdist in dist/, found ${#sdists[@]}"
printf 'built wheel: %s\n' "${wheels[0]##*/}"
printf 'built sdist: %s\n' "${sdists[0]##*/}"

# --- 2/5: metadata ---------------------------------------------------------
# This is the only step whose *tooling* is fetched from an index: `uvx` resolves
# twine on first use (later runs hit the cache). The wheel's own runtime
# dependencies in step 3 come from the same configured index / cache, and no
# other step touches the network.
step "2/5 Check distribution metadata (uvx twine check dist/*)"
uvx twine check dist/*

# --- 3/5: clean venv, exact wheel -----------------------------------------
step "3/5 Recreate clean venv at $VENV and install the exact wheel"
rm -rf "$VENV"
uv venv "$VENV"
uv pip install --python "$VENV/bin/python" "${wheels[0]}"

# --- 4/5: import and version consistency -----------------------------------
step "4/5 Import eyas in the clean venv and check the CLI version"
module_path="$("$VENV/bin/python" -c 'import eyas; print(eyas.__file__)')"
printf 'import eyas OK: %s\n' "$module_path"
# Guard against a stray editable/checkout import: the module must live inside
# the throwaway venv, i.e. it came from the built wheel.
case "$module_path" in
  "$VENV"/*) ;;
  *) fail "eyas imported from outside the venv: $module_path" ;;
esac
# The CLI's reported version must equal the built distribution's metadata.
# This catches a hardcoded __version__ drifting from pyproject.toml: twine
# checks metadata only, so without this the wheel could install cleanly while
# `eyas --version` advertised the wrong release.
metadata_version="$(
  "$VENV/bin/python" -c 'import importlib.metadata; print(importlib.metadata.version("eyas"))'
)"
cli_version="$("$VENV/bin/eyas" --version)"
printf 'distribution metadata version: %s\n' "$metadata_version"
printf 'eyas --version: %s\n' "$cli_version"
[[ "$cli_version" == "eyas $metadata_version" ]] ||
  fail "version mismatch: CLI says '$cli_version', metadata says '$metadata_version'"

# --- 5/5: console script ---------------------------------------------------
step "5/5 Run eyas --help from the clean venv"
"$VENV/bin/eyas" --help

printf '\n==> ALL PREFLIGHT CHECKS PASSED\n'
