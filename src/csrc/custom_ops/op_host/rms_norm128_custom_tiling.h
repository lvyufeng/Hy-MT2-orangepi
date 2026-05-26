#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNorm128CustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNorm128Custom, RmsNorm128CustomTilingData)
}  // namespace optiling
