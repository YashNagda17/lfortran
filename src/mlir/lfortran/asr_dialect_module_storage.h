// Module/context-owned ASR dialect side storage (V1).
//
// Holds structural relationships that must not appear as pointer-shaped MLIR
// attributes in the default dump: child op refs, statement bodies, op
// sequences, and ASR type metadata for pretty printing.
#pragma once

#include <mlir_api.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void ASR_ModuleStorageInit(MLIR_Context *ctx);
void ASR_ModuleStorageClear(MLIR_Context *ctx);

void ASR_ModuleStorageSetFieldOp(
    MLIR_OpHandle parent, const char *field, MLIR_OpHandle child);
MLIR_OpHandle ASR_ModuleStorageGetFieldOp(
    MLIR_OpHandle parent, const char *field);

void ASR_ModuleStorageSetFieldOpSeq(
    MLIR_OpHandle parent, const char *field,
    MLIR_OpHandle *ops, size_t n);
MLIR_OpHandle *ASR_ModuleStorageGetFieldOpSeq(
    MLIR_OpHandle parent, const char *field, size_t *n_out);

void ASR_ModuleStorageSetBody(
    MLIR_OpHandle parent, MLIR_OpHandle *stmts, size_t n);
size_t ASR_ModuleStorageBodyCount(MLIR_OpHandle parent);
MLIR_OpHandle ASR_ModuleStorageBodyOp(MLIR_OpHandle parent, size_t index);

void ASR_ModuleStorageSetTypeInfo(
    MLIR_TypeHandle ty, int64_t asr_kind, bool is_array, int64_t array_len);
bool ASR_ModuleStorageGetTypeInfo(
    MLIR_TypeHandle ty, int64_t *asr_kind_out,
    bool *is_array_out, int64_t *array_len_out);

#ifdef __cplusplus
}
#endif
