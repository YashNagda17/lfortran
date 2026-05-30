# MLIR C API for mlir-new: Initial ASR -> high-level MLIR -> LLVM dialect -> LLVM IR.
#
# Upstream certik/mlir lives in upstream/ (git submodule, recursive: corec,
# corec-stdlib). Only the mlir-new pipeline sources are compiled here; the rest
# of upstream/ is kept intact for easy merges. LFortran-specific overrides
# (hosted corec platform, printf vector.print lowering, etc.) live in lfortran/.

if (NOT WITH_MLIR OR NOT WITH_LLVM)
    return()
endif()

set(LF_MLIR_UPSTREAM ${CMAKE_CURRENT_SOURCE_DIR}/upstream)
set(LF_MLIR_LFORTRAN ${CMAKE_CURRENT_SOURCE_DIR}/lfortran)

function(lf_mlir_source out_var relpath)
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
    lf_mlir_source(abs ${rel})
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

# mlir-new pipeline only (not parser, tinyc, macho/wmir backends, etc.)
set(LF_MLIR_API_C_REL
    mlir_op_names.c
    mlir_lift_cf_to_scf.c
    mlir_lower_to_llvm.c
    mlir_translate_to_llvm_ir.c
    mlir_translate_to_wasm.c
    mlir_generic_printer.c
    mlir_classic_printer.c
    mlir_llvm_to_wasmssa.c
    mlir_wasmssa_to_wasmstack.c
    mlir_wasmstack_to_bin.c
    mlir_wasm_to_wat.c
)
set(LF_MLIR_API_C_SRC "")
foreach (rel IN LISTS LF_MLIR_API_C_REL)
    lf_mlir_source(abs ${rel})
    list(APPEND LF_MLIR_API_C_SRC ${abs})
endforeach()

lf_mlir_source(LF_MLIR_API_IMPL_CPP mlir_api_impl_upstream.cpp)

add_library(lfortran_mlir_c_api STATIC
    ${LF_MLIR_API_C_SRC}
    ${LF_MLIR_API_IMPL_CPP}
)
target_include_directories(lfortran_mlir_c_api BEFORE PUBLIC
    ${LF_MLIR_LFORTRAN}
    ${LF_MLIR_UPSTREAM}
    ${LF_MLIR_LFORTRAN}/corec
    ${LF_MLIR_UPSTREAM}/corec
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
