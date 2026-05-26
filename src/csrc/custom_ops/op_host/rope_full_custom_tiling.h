#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RopeFullCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF(uint32_t, halfDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RopeFullCustom, RopeFullCustomTilingData)
}  // namespace optiling
