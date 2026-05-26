cmake_minimum_required(VERSION 3.20)
project(hy_mt2_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Ascend toolkit root. This Orange Pi AIPro 20T install lives at a
# non-standard path — the usual `ascend-toolkit/latest` symlink isn't
# present. Override with -DASCEND_TOOLKIT_ROOT=... if your install
# differs.
set(ASCEND_TOOLKIT_ROOT "/usr/local/Ascend/cann-8.5.0/aarch64-linux"
    CACHE PATH "Ascend toolkit root")

# Where the custom_opp .run installer extracts. Defaults to the location
# scripts/install_custom_ops.sh uses; override with -DCUSTOM_OPP_VENDOR=...
set(CUSTOM_OPP_VENDOR "${CMAKE_SOURCE_DIR}/custom_opp_install/vendors/customize"
    CACHE PATH "Custom OPP vendor root (output of custom_opp .run installer)")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Aggressive optimization for the A55 cluster.
add_compile_options(-O3 -ffast-math -march=armv8.2-a+fp16 -Wall -Wextra
                    -Wno-unused-parameter -Wno-missing-field-initializers)

add_library(hy_mt2_engine
  src/csrc/lib/acl_context.cpp
  src/csrc/lib/tensor.cpp
)

target_include_directories(hy_mt2_engine PUBLIC
  ${CMAKE_SOURCE_DIR}/src/csrc/include
  ${ASCEND_TOOLKIT_ROOT}/include
)

target_link_directories(hy_mt2_engine PUBLIC
  ${ASCEND_TOOLKIT_ROOT}/lib64
)

# Custom OPP is optional at the Phase 0 scaffold stage — only required
# once the engine actually invokes a custom kernel.
if(EXISTS "${CUSTOM_OPP_VENDOR}/op_api/include")
  target_include_directories(hy_mt2_engine PUBLIC "${CUSTOM_OPP_VENDOR}/op_api/include")
  target_link_directories(hy_mt2_engine PUBLIC   "${CUSTOM_OPP_VENDOR}/op_api/lib")
  target_link_libraries(hy_mt2_engine PUBLIC cust_opapi)
  target_link_options(hy_mt2_engine PUBLIC "-Wl,-rpath,${CUSTOM_OPP_VENDOR}/op_api/lib")
endif()

target_link_libraries(hy_mt2_engine PUBLIC
  ascendcl
  nnopbase
  opapi
)

# Per-file test / bench targets — glob so adding tests/test_foo.cpp or
# bench/bench_foo.cpp picks up automatically on the next cmake re-config.
file(GLOB TEST_SRCS  CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
file(GLOB BENCH_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/bench/*.cpp)
file(GLOB TOOL_SRCS  CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/csrc/tools/*.cpp)

foreach(src ${TEST_SRCS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE hy_mt2_engine)
endforeach()

foreach(src ${BENCH_SRCS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE hy_mt2_engine)
endforeach()

foreach(src ${TOOL_SRCS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE hy_mt2_engine)
endforeach()
