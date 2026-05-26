#!/usr/bin/env bash
# Build and install the AscendC custom operators under
# $REPO_ROOT/custom_opp_install/ (or $CUSTOM_OPP_INSTALL_DIR if set).
#
# The installed location matches the default `CUSTOM_OPP_VENDOR` in the
# root CMakeLists and the `ASCEND_CUSTOM_OPP_PATH` exported by
# scripts/set_env.sh, so engine builds pick it up automatically.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUSTOM_OPS_DIR="$REPO_ROOT/src/csrc/custom_ops"
INSTALL_DIR="${CUSTOM_OPP_INSTALL_DIR:-$REPO_ROOT/custom_opp_install}"

# Phase 0 has no custom op source yet — bail gracefully so contributors
# can still source set_env.sh and run cmake before the kernel work lands.
if [[ ! -f "$CUSTOM_OPS_DIR/build.sh" ]]; then
    echo "No custom_ops/build.sh yet — Phase 2 ports the kernels."
    exit 0
fi

# The Orange Pi image ships stale ASCEND_* variables pointing at the
# non-existent ascend-toolkit/laster path, so force the known-good CANN root.
DEFAULT_CANN_ROOT="${HY_MT2_CANN_ROOT:-/usr/local/Ascend/cann-8.5.0}"
export ASCEND_HOME_PATH="$DEFAULT_CANN_ROOT"
export ASCEND_AICPU_PATH="$DEFAULT_CANN_ROOT"
export ASCEND_OPP_PATH="$DEFAULT_CANN_ROOT/opp"
export BASE_LIBS_PATH="$DEFAULT_CANN_ROOT"
export LD_LIBRARY_PATH="$DEFAULT_CANN_ROOT/lib64:$DEFAULT_CANN_ROOT/aarch64-linux/lib64:$DEFAULT_CANN_ROOT/aarch64-linux/devlib:${LD_LIBRARY_PATH:-}"
export PATH="$DEFAULT_CANN_ROOT/bin:$DEFAULT_CANN_ROOT/compiler/ccec_compiler/bin:${PATH:-}"

cd "$CUSTOM_OPS_DIR"
./build.sh

RUN_PKG="$CUSTOM_OPS_DIR/build_out/custom_opp_ubuntu_aarch64.run"
if [[ ! -f "$RUN_PKG" ]]; then
  echo "ERROR: $RUN_PKG was not produced by build.sh" >&2
  exit 1
fi
mkdir -p "$INSTALL_DIR"
"$RUN_PKG" --install-path="$INSTALL_DIR" --quiet
echo "Custom ops installed at $INSTALL_DIR"
