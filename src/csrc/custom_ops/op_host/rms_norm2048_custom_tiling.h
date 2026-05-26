#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNorm2048CustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNorm2048Custom, RmsNorm2048CustomTilingData)
}  // namespace optiling
