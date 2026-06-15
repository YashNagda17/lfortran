# MLIR C API for mlir-new: final ASR -> LLVM dialect MLIR -> LLVM IR.
#
# Two backends (selected via --backend=mlir-new-native|mlir-new-upstream):
#   native   — mlir_api_impl.c + native print/translate
#   upstream — mlir_api_impl_upstream.cpp (prefixed core symbols) + upstream print/translate
#
# LFortran-specific hooks live in lfortran/mlir_lfortran_hooks.{c,h}.

if (NOT WITH_MLIR OR NOT WITH_LLVM)
    return()
endif()

set(LF_MLIR_UPSTREAM ${CMAKE_CURRENT_SOURCE_DIR}/upstream)
set(LF_MLIR_LFORTRAN ${CMAKE_CURRENT_SOURCE_DIR}/lfortran)

function(lf_corec_source out_var relpath)
    if (EXISTS "${LF_MLIR_LFORTRAN}/${relpath}")
        set(${out_var} "${LF_MLIR_LFORTRAN}/${relpath}" PARENT_SCOPE)
    else()
        set(${out_var} "${LF_MLIR_UPSTREAM}/${relpath}" PARENT_SCOPE)
    endif()
endfunction()

if (NOT EXISTS "${LF_MLIR_UPSTREAM}/mlir_api.h")
    message(FATAL_ERROR
        "src/mlir/upstream is missing. Initialize submodules with:\n"
        "  git submodule update --init --recursive src/mlir/upstream")
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
set(LF_COREC_BASE_SRC_ABS "")
foreach (rel IN LISTS LF_COREC_BASE_SRC)
    lf_corec_source(abs ${rel})
    list(APPEND LF_COREC_BASE_SRC_ABS ${abs})
endforeach()

add_library(lfortran_corec STATIC ${LF_COREC_BASE_SRC_ABS})
target_include_directories(lfortran_corec BEFORE PUBLIC
    ${LF_MLIR_LFORTRAN}/corec
    ${LF_MLIR_UPSTREAM}/corec
)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(lfortran_corec PRIVATE
        ${LF_MLIR_LFORTRAN}/corec/platform/platform_linux.c)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_sources(lfortran_corec PRIVATE
        ${LF_MLIR_LFORTRAN}/corec/platform/platform_macos.c)
elseif (WIN32)
    target_sources(lfortran_corec PRIVATE
        ${LF_MLIR_LFORTRAN}/corec/platform/platform_windows.c)
else()
    message(FATAL_ERROR "lfortran_corec: add corec platform file for ${CMAKE_SYSTEM_NAME}")
endif()
target_compile_definitions(lfortran_corec PRIVATE PLATFORM_SKIP_ENTRY PLATFORM_HOSTED)

# Shared native pipeline + core API + backend dispatch.
set(LF_MLIR_PIPELINE_SRC
    ${LF_MLIR_UPSTREAM}/mlir_op_names.c
    ${LF_MLIR_UPSTREAM}/mlir_api_impl.c
    ${LF_MLIR_UPSTREAM}/mlir_lift_cf_to_scf.c
    ${LF_MLIR_LFORTRAN}/mlir_lower_to_llvm_lfortran.c
    ${LF_MLIR_LFORTRAN}/mlir_translate_to_llvm_ir_lfortran.c
    ${LF_MLIR_UPSTREAM}/mlir_generic_printer.c
    ${LF_MLIR_UPSTREAM}/mlir_classic_printer.c
    ${LF_MLIR_LFORTRAN}/mlir_lfortran_hooks.c
    ${LF_MLIR_LFORTRAN}/mlir_new_backend.c
)

add_library(lfortran_mlir_pipeline STATIC ${LF_MLIR_PIPELINE_SRC})
target_include_directories(lfortran_mlir_pipeline BEFORE PUBLIC
    ${LF_MLIR_UPSTREAM}
    ${LF_MLIR_LFORTRAN}
    ${LF_MLIR_LFORTRAN}/corec
    ${LF_MLIR_UPSTREAM}/corec
)
target_link_libraries(lfortran_mlir_pipeline PUBLIC lfortran_corec)

# Upstream core API with MLIR_Upstream_* symbol prefix on shared entry points.
add_library(lfortran_mlir_upstream_impl STATIC
    ${LF_MLIR_UPSTREAM}/mlir_api_impl_upstream.cpp
)
target_include_directories(lfortran_mlir_upstream_impl BEFORE PUBLIC
    ${LF_MLIR_UPSTREAM}
    ${LF_MLIR_LFORTRAN}
    ${LF_MLIR_LFORTRAN}/corec
    ${LF_MLIR_UPSTREAM}/corec
)
target_include_directories(lfortran_mlir_upstream_impl PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
if (DEFINED LFORTRAN_CXX_NO_RTTI_FLAG)
    target_compile_options(lfortran_mlir_upstream_impl PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${LFORTRAN_CXX_NO_RTTI_FLAG}>)
endif()
target_compile_options(lfortran_mlir_upstream_impl PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-include${LF_MLIR_LFORTRAN}/mlir_upstream_api_rename.h>)
if (LFORTRAN_MLIR_WASM_LOWERING OR WITH_TARGET_WASM)
    target_compile_definitions(lfortran_mlir_upstream_impl PRIVATE
        LFORTRAN_MLIR_WASM_LOWERING=1)
else()
    target_sources(lfortran_mlir_upstream_impl PRIVATE
        ${LF_MLIR_LFORTRAN}/mlir_api_impl_upstream_wasm_stubs.c)
endif()
# Do not PUBLIC-link pipeline here: CMake would place upstream_impl before
# pipeline on the final link line, while mlir_new_backend.c (in pipeline)
# references MLIR_Upstream_* from upstream_impl.
target_link_libraries(lfortran_mlir_upstream_impl PUBLIC
    p::mlir
    p::llvm
)

# Optional native wasm pipeline (tinyc); host mlir-new builds omit this.
set(LF_MLIR_WASM_C_SRC
    ${LF_MLIR_UPSTREAM}/mlir_translate_to_wasm.c
    ${LF_MLIR_UPSTREAM}/mlir_llvm_to_wasmssa.c
    ${LF_MLIR_UPSTREAM}/mlir_wasmssa_to_wasmstack.c
    ${LF_MLIR_UPSTREAM}/mlir_wasmstack_to_bin.c
    ${LF_MLIR_UPSTREAM}/mlir_wasm_to_wat.c
)

if (LFORTRAN_MLIR_WASM_LOWERING OR WITH_TARGET_WASM)
    add_library(lfortran_mlir_wasm_lowering STATIC ${LF_MLIR_WASM_C_SRC})
    target_include_directories(lfortran_mlir_wasm_lowering BEFORE PUBLIC
        ${LF_MLIR_UPSTREAM}
        ${LF_MLIR_LFORTRAN}
        ${LF_MLIR_LFORTRAN}/corec
        ${LF_MLIR_UPSTREAM}/corec
    )
    target_link_libraries(lfortran_mlir_wasm_lowering PUBLIC lfortran_mlir_pipeline)
endif()

add_library(lfortran_mlir_c_api INTERFACE)
# pipeline <-> upstream_impl is circular (vtable vs MLIR_LiftCfToScfNative).
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
    target_link_libraries(lfortran_mlir_c_api INTERFACE
        "-Wl,--start-group"
        lfortran_mlir_pipeline
        lfortran_mlir_upstream_impl
        "-Wl,--end-group"
    )
else()
    target_link_libraries(lfortran_mlir_c_api INTERFACE
        lfortran_mlir_pipeline
        lfortran_mlir_upstream_impl
    )
endif()
if (LFORTRAN_MLIR_WASM_LOWERING OR WITH_TARGET_WASM)
    target_link_libraries(lfortran_mlir_c_api INTERFACE lfortran_mlir_wasm_lowering)
endif()
