#ifndef ASR_DIALECT_EMIT_REGISTRY_H
#define ASR_DIALECT_EMIT_REGISTRY_H

#include <mlir_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ASR_DialectEmitRegistryClear(void);
void ASR_DialectEmitRegistryAddDoLoopBody(
    MLIR_OpHandle do_op, MLIR_OpHandle *stmts, size_t n_stmts);
size_t ASR_DialectEmitRegistryDoLoopBodyCount(MLIR_OpHandle do_op);
MLIR_OpHandle ASR_DialectEmitRegistryDoLoopBodyOp(
    MLIR_OpHandle do_op, size_t index);

#ifdef __cplusplus
}
#endif

#endif
