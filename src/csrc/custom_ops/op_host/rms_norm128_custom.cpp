#include "rms_norm128_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    RmsNorm128CustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const auto& shape = xShape->GetStorageShape();
    const uint32_t rows = static_cast<uint32_t>(shape.GetDim(0));

    const auto* attrs = context->GetAttrs();
    const float* epsAttr = attrs->GetAttrPointer<float>(0);

    tiling.set_rows(rows);
    tiling.set_epsilon(epsAttr ? *epsAttr : 1.0e-5f);

    context->SetBlockDim(rows);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class RmsNorm128Custom : public OpDef {
public:
    explicit RmsNorm128Custom(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("epsilon").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(RmsNorm128Custom);
}  // namespace ops
