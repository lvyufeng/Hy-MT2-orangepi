#!/usr/bin/env bash
# Source this script before running any engine binary. Sets the Ascend
# toolkit and custom_opp paths.
#
# The CANN install on this Orange Pi AIPro 20T lives at a non-standard
# path; the usual `ascend-toolkit/latest` symlink is absent. Override
# with HY_MT2_ASCEND_TOOLKIT_ROOT / HY_MT2_CUSTOM_OPP_VENDOR if your
# install differs.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export ASCEND_TOOLKIT_ROOT="${HY_MT2_ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/cann-8.5.0/aarch64-linux}"
export LD_LIBRARY_PATH="$ASCEND_TOOLKIT_ROOT/lib64:${LD_LIBRARY_PATH:-}"

CUSTOM_OPP_VENDOR="${HY_MT2_CUSTOM_OPP_VENDOR:-$REPO_ROOT/custom_opp_install/vendors/customize}"
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_VENDOR:${ASCEND_CUSTOM_OPP_PATH:-}"
