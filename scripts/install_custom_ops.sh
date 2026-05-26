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

# The AscendC build script expects ASCEND_HOME_PATH (or ASCEND_AICPU_PATH /
# BASE_LIBS_PATH). Source the CANN env script if available and the user
# hasn't already exported it.
if [[ -z "${ASCEND_HOME_PATH:-}" && -z "${ASCEND_AICPU_PATH:-}" && -z "${BASE_LIBS_PATH:-}" ]]; then
    DEFAULT_CANN_ROOT="${HY_MT2_ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/cann-8.5.0/aarch64-linux}"
    # The setenv.bash lives one directory up from the platform-specific
    # subdir (i.e. at /usr/local/Ascend/cann-8.5.0/bin/setenv.bash on this
    # install).
    CANN_PARENT="$(dirname "$DEFAULT_CANN_ROOT")"
    if [[ -f "$CANN_PARENT/bin/setenv.bash" ]]; then
        # shellcheck disable=SC1090
        source "$CANN_PARENT/bin/setenv.bash"
    else
        export ASCEND_HOME_PATH="$DEFAULT_CANN_ROOT"
    fi
fi

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
