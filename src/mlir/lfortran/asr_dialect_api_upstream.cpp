// Upstream ASR dialect API stub — delegates to native until ODS registration lands.
#include "asr_dialect_api.h"

extern MLIR_OpHandle ASR_DialectCreateOpNative(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields);
extern ASR_DialectOpKind ASR_DialectGetOpKindNative(MLIR_OpHandle op);
extern bool ASR_DialectVerifyNative(MLIR_Context *ctx, MLIR_OpHandle module);
extern bool ASR_DialectLowerToHighMLIRNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options);
extern string ASR_DialectPrintNative(MLIR_Context *ctx, MLIR_OpHandle module);

MLIR_OpHandle ASR_DialectCreateOpUpstream(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields) {
    return ASR_DialectCreateOpNative(ctx, kind, loc, fields, n_fields);
}

ASR_DialectOpKind ASR_DialectGetOpKindUpstream(MLIR_OpHandle op) {
    return ASR_DialectGetOpKindNative(op);
}

bool ASR_DialectVerifyUpstream(MLIR_Context *ctx, MLIR_OpHandle module) {
    return ASR_DialectVerifyNative(ctx, module);
}

bool ASR_DialectLowerToHighMLIRUpstream(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options) {
    return ASR_DialectLowerToHighMLIRNative(ctx, module, options);
}

string ASR_DialectPrintUpstream(MLIR_Context *ctx, MLIR_OpHandle module) {
    return ASR_DialectPrintNative(ctx, module);
}
