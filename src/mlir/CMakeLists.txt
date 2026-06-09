# MLIR C API for mlir-new: Initial ASR -> high-level MLIR -> LLVM dialect -> LLVM IR.
#
# Upstream certik/mlir lives in upstream/ (git submodule, recursive: corec,
# corec-stdlib). Only the mlir-new pipeline sources are compiled here; the rest
# of upstream/ is kept intact for easy merges. LFortran-specific hooks
# (hosted corec platform, printf vector.print, memref<1xi32>, etc.) live in
# lfortran/.
#
# Link surface:
#   lfortran_mlir_c_api_core     — ASR dialect -> high MLIR -> LLVM IR (native)
#   lfortran_mlir_upstream_shim  — optional upstream MLIR/LLVM API backend
#   lfortran_mlir_wasm_lowering  — optional native wasm pipeline (tinyc)
#   lfortran_asr_dialect         — ASR dialect C API + lowering handlers
#   lfortran_mlir_c_api          — INTERFACE aggregate used by libasr

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

# mlir-new host pipeline (no native wasm translation chain).
set(LF_MLIR_API_C_CORE_REL
    mlir_op_names.c
    mlir_lift_cf_to_scf.c
    mlir_lower_to_llvm.c
    mlir_translate_to_llvm_ir.c
    mlir_generic_printer.c
    mlir_classic_printer.c
)
set(LF_MLIR_API_C_CORE_SRC "")
foreach (rel IN LISTS LF_MLIR_API_C_CORE_REL)
    lf_mlir_source(abs ${rel})
    list(APPEND LF_MLIR_API_C_CORE_SRC ${abs})
endforeach()

set(LF_MLIR_WASM_C_REL
    mlir_translate_to_wasm.c
    mlir_llvm_to_wasmssa.c
    mlir_wasmssa_to_wasmstack.c
    mlir_wasmstack_to_bin.c
    mlir_wasm_to_wat.c
)
set(LF_MLIR_WASM_C_SRC "")
foreach (rel IN LISTS LF_MLIR_WASM_C_REL)
    lf_mlir_source(abs ${rel})
    list(APPEND LF_MLIR_WASM_C_SRC ${abs})
endforeach()

set(LF_MLIR_LFORTRAN_HOOKS
    mlir_lfortran_hooks.c
)
set(LF_MLIR_LFORTRAN_HOOKS_ABS "")
foreach (rel IN LISTS LF_MLIR_LFORTRAN_HOOKS)
    list(APPEND LF_MLIR_LFORTRAN_HOOKS_ABS ${LF_MLIR_LFORTRAN}/${rel})
endforeach()

if (EXISTS "${LF_MLIR_LFORTRAN}/mlir_api_impl_upstream.cpp")
    set(LF_MLIR_API_IMPL_CPP ${LF_MLIR_LFORTRAN}/mlir_api_impl_upstream.cpp)
else()
    set(LF_MLIR_API_IMPL_CPP ${LF_MLIR_UPSTREAM}/mlir_api_impl_upstream.cpp)
endif()

lf_mlir_source(LF_MLIR_LFORTRAN_HOOKS_SRC mlir_lfortran_hooks.c)
add_library(lfortran_mlir_c_api_core STATIC
    ${LF_MLIR_API_C_CORE_SRC}
    ${LF_MLIR_LFORTRAN_HOOKS_SRC}
)
target_include_directories(lfortran_mlir_c_api_core BEFORE PUBLIC
    ${LF_MLIR_LFORTRAN}
    ${LF_MLIR_UPSTREAM}
    ${LF_MLIR_LFORTRAN}/corec
    ${LF_MLIR_UPSTREAM}/corec
)
target_link_libraries(lfortran_mlir_c_api_core PUBLIC lfortran_corec)

add_library(lfortran_mlir_upstream_shim STATIC
    ${LF_MLIR_API_IMPL_CPP}
)
target_include_directories(lfortran_mlir_upstream_shim BEFORE PUBLIC
    ${LF_MLIR_LFORTRAN}
    ${LF_MLIR_UPSTREAM}
    ${LF_MLIR_LFORTRAN}/corec
    ${LF_MLIR_UPSTREAM}/corec
)
target_include_directories(lfortran_mlir_upstream_shim PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
if (DEFINED LFORTRAN_CXX_NO_RTTI_FLAG)
    target_compile_options(lfortran_mlir_upstream_shim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${LFORTRAN_CXX_NO_RTTI_FLAG}>)
endif()
target_link_libraries(lfortran_mlir_upstream_shim PUBLIC
    lfortran_mlir_c_api_core
    p::mlir
    p::llvm
)
if (LFORTRAN_MLIR_WASM_LOWERING OR WITH_TARGET_WASM)
    target_compile_definitions(lfortran_mlir_upstream_shim PRIVATE
        LFORTRAN_MLIR_WASM_LOWERING=1)
endif()

if (LFORTRAN_MLIR_WASM_LOWERING OR WITH_TARGET_WASM)
    add_library(lfortran_mlir_wasm_lowering STATIC ${LF_MLIR_WASM_C_SRC})
    target_include_directories(lfortran_mlir_wasm_lowering BEFORE PUBLIC
        ${LF_MLIR_LFORTRAN}
        ${LF_MLIR_UPSTREAM}
        ${LF_MLIR_LFORTRAN}/corec
        ${LF_MLIR_UPSTREAM}/corec
    )
    target_link_libraries(lfortran_mlir_wasm_lowering PUBLIC
        lfortran_mlir_c_api_core
    )
endif()

add_library(lfortran_mlir_c_api INTERFACE)
target_link_libraries(lfortran_mlir_c_api INTERFACE
    lfortran_mlir_c_api_core
    lfortran_mlir_upstream_shim
)
if (LFORTRAN_MLIR_WASM_LOWERING OR WITH_TARGET_WASM)
    target_link_libraries(lfortran_mlir_c_api INTERFACE lfortran_mlir_wasm_lowering)
endif()

# ---------------------------------------------------------------------------
# ASR dialect (Design A): generated from ASR.asdl + native C API
# ---------------------------------------------------------------------------

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LF_ASR_ASDL ${CMAKE_CURRENT_SOURCE_DIR}/../libasr/ASR.asdl)
set(LF_ASR_DIALECT_GEN ${CMAKE_CURRENT_SOURCE_DIR}/../libasr/asdl_to_asr_dialect.py)
set(LF_ASR_DIALECT_MLIR_GEN ${CMAKE_CURRENT_SOURCE_DIR}/generated)
set(LF_ASR_DIALECT_CODEGEN ${CMAKE_CURRENT_SOURCE_DIR}/../libasr/codegen)

add_custom_command(
    OUTPUT
        ${LF_ASR_DIALECT_MLIR_GEN}/asr_dialect_schema.h
        ${LF_ASR_DIALECT_MLIR_GEN}/asr_dialect_api_generated.h
        ${LF_ASR_DIALECT_MLIR_GEN}/asr_dialect_lowering_dispatch.h
        ${LF_ASR_DIALECT_MLIR_GEN}/asr_dialect_print_helpers.h
    COMMAND ${Python3_EXECUTABLE} ${LF_ASR_DIALECT_GEN}
        --asdl ${LF_ASR_ASDL}
        --out-mlir-dir ${LF_ASR_DIALECT_MLIR_GEN}
        --out-codegen-dir ${LF_ASR_DIALECT_CODEGEN}
        --policy ${LF_MLIR_LFORTRAN}/asr_dialect_storage_policy.h
    DEPENDS ${LF_ASR_ASDL} ${LF_ASR_DIALECT_GEN}
        ${LF_MLIR_LFORTRAN}/asr_dialect_storage_policy.h
    COMMENT "Generating ASR dialect artifacts from ASR.asdl"
)

add_custom_target(generate_asr_dialect
    DEPENDS
        ${LF_ASR_DIALECT_MLIR_GEN}/asr_dialect_schema.h
        ${LF_ASR_DIALECT_MLIR_GEN}/asr_dialect_api_generated.h
        ${LF_ASR_DIALECT_MLIR_GEN}/asr_dialect_lowering_dispatch.h
        ${LF_ASR_DIALECT_MLIR_GEN}/asr_dialect_print_helpers.h
)

add_library(lfortran_asr_dialect STATIC
    lfortran/asr_dialect_api.c
    lfortran/asr_dialect_api_native.c
    lfortran/asr_dialect_pretty_print.c
    lfortran/asr_dialect_lowering_handlers.c
)
add_dependencies(lfortran_asr_dialect generate_asr_dialect)
target_include_directories(lfortran_asr_dialect BEFORE PUBLIC
    ${LF_MLIR_LFORTRAN}
    ${LF_MLIR_UPSTREAM}
    ${LF_MLIR_LFORTRAN}/corec
    ${LF_MLIR_UPSTREAM}/corec
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(lfortran_asr_dialect PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(lfortran_asr_dialect PUBLIC
    lfortran_mlir_c_api
    lfortran_corec
)
