#include "hy_mt2/acl_context.h"

namespace hy_mt2 {

AclContext::AclContext(int device_id) : device_id_(device_id) {
    check_acl(aclInit(nullptr), "aclInit");
    initialized_ = true;
    check_acl(aclrtSetDevice(device_id_), "aclrtSetDevice");
    device_set_ = true;
    check_acl(aclrtCreateStream(&stream_), "aclrtCreateStream");
}

AclContext::~AclContext() {
    if (stream_ != nullptr) {
        aclrtDestroyStream(stream_);
    }
    if (device_set_) {
        aclrtResetDevice(device_id_);
    }
    if (initialized_) {
        aclFinalize();
    }
}

}  // namespace hy_mt2
