# Minimal MLIR C API for mlir-new: ASR -> LLVM-dialect MLIR -> LLVM IR text.

if (NOT WITH_MLIR OR NOT WITH_LLVM)
    return()
endif()

set(LF_COREC_BASE_SRC
    corec/base/io.c
    corec/base/buddy.c
    corec/base/arena.c
    corec/base/scratch.c
    corec/base/strbuf.c
    corec/base/format.c
    corec/base/math.c
    corec/base/string.c
    corec/base/mem.c
    corec/base/numconv.c
    corec/base/assert.c
    corec/base/exit.c
)

add_library(lfortran_corec STATIC ${LF_COREC_BASE_SRC})
target_include_directories(lfortran_corec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/corec
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(lfortran_corec PRIVATE corec/platform/platform_linux.c)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_sources(lfortran_corec PRIVATE corec/platform/platform_macos.c)
elseif (WIN32)
    target_sources(lfortran_corec PRIVATE corec/platform/platform_windows.c)
else()
    message(FATAL_ERROR "lfortran_corec: add corec platform file for ${CMAKE_SYSTEM_NAME}")
endif()
target_compile_definitions(lfortran_corec PRIVATE PLATFORM_SKIP_ENTRY PLATFORM_HOSTED)

set(LF_MLIR_API_C_SRC
    mlir_op_names.c
    mlir_translate_to_llvm_ir.c
    mlir_lift_cf_to_scf_stub.c
)

add_library(lfortran_mlir_c_api STATIC
    ${LF_MLIR_API_C_SRC}
    mlir_api_impl_upstream.cpp
)
target_include_directories(lfortran_mlir_c_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/corec
)
target_include_directories(lfortran_mlir_c_api PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
if (DEFINED LFORTRAN_CXX_NO_RTTI_FLAG)
    target_compile_options(lfortran_mlir_c_api PRIVATE ${LFORTRAN_CXX_NO_RTTI_FLAG})
endif()

target_link_libraries(lfortran_mlir_c_api PUBLIC
    lfortran_corec
    p::mlir
    p::llvm
)
