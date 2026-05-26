// Translate-only build: mlir_api_impl_upstream.cpp references
// MLIR_LiftCfToScfNative but the full mlir_lift_cf_to_scf.c pass is not
// needed when emitting llvm-dialect MLIR directly (no cf->scf lowering).
#include <mlir_api.h>

bool MLIR_LiftCfToScfNative(MLIR_Context *ctx, MLIR_OpHandle module) {
    (void)ctx;
    (void)module;
    return true;
}
