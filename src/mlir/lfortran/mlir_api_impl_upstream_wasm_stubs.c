// Stubs for upstream wasm entry points when LFORTRAN_MLIR_WASM_LOWERING is off.
// Real implementations come from upstream/mlir_api_impl_upstream.cpp when
// wasm lowering is enabled.

#include <stdio.h>

#include "mlir_api.h"

bool MLIR_LowerToLLVMDialectForWasmUpstream(MLIR_Context *ctx,
                                            MLIR_OpHandle module_h) {
    (void)ctx;
    (void)module_h;
    fprintf(stderr,
            "MLIR_LowerToLLVMDialectForWasmUpstream: WebAssembly lowering is "
            "disabled (reconfigure with -DLFORTRAN_MLIR_WASM_LOWERING=yes)\n");
    return false;
}

string MLIR_TranslateModuleToWasmUpstream(MLIR_Context *ctx,
                                          MLIR_OpHandle module_h) {
    (void)ctx;
    (void)module_h;
    fprintf(stderr,
            "MLIR_TranslateModuleToWasmUpstream: WebAssembly lowering is "
            "disabled (reconfigure with -DLFORTRAN_MLIR_WASM_LOWERING=yes)\n");
    string result = {0};
    return result;
}
