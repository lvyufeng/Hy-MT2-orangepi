#!/usr/bin/env bash
# Build the engine into $REPO_ROOT/build (or $BUILD_DIR if set).
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
fi

mkdir -p "$BUILD_DIR"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" "$@"
cmake --build "$BUILD_DIR" -j"$(nproc)"
