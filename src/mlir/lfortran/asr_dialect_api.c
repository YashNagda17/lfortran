// ASR dialect API dispatch — native backend only.
// Future: add lfortran/asr_dialect_api_upstream.cpp with real upstream ODS ASR dialect
// ops and branch here on USE_MLIR_Upstream (same pattern as mlir_api_impl_upstream.cpp).
#include "asr_dialect_api.h"
#include "asr_dialect_storage.h"

#include <stdio.h>
#include <string.h>

// Native backend (always linked).
extern MLIR_OpHandle ASR_DialectCreateOpNative(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields);
extern ASR_DialectOpKind ASR_DialectGetOpKindNative(MLIR_OpHandle op);
extern bool ASR_DialectVerifyNative(MLIR_Context *ctx, MLIR_OpHandle module);
extern void ASR_DialectClearVerifyErrorNative(void);
extern const ASR_DialectVerifyError *ASR_DialectGetLastVerifyErrorNative(void);
extern void ASR_DialectClearCodegenErrorNative(void);
extern const ASR_DialectCodegenError *ASR_DialectGetLastCodegenErrorNative(void);
extern bool ASR_DialectLowerToHighMLIRNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options);
extern string ASR_DialectPrintNative(MLIR_Context *ctx, MLIR_OpHandle module);

MLIR_OpHandle ASR_DialectCreateOp(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields) {
    // Future: ASR_DialectCreateOpUpstream from asr_dialect_api_upstream.cpp
    return ASR_DialectCreateOpNative(ctx, kind, loc, fields, n_fields);
}

ASR_DialectOpKind ASR_DialectGetOpKind(MLIR_OpHandle op) {
    // Future: ASR_DialectGetOpKindUpstream from asr_dialect_api_upstream.cpp
    return ASR_DialectGetOpKindNative(op);
}

const ASR_DialectOpSchema *ASR_DialectLookupSchema(ASR_DialectOpKind kind) {
    for (size_t i = 0; i < ASR_DIALECT_SCHEMA_COUNT; ++i) {
        if (ASR_DIALECT_SCHEMA[i].kind == kind) {
            return &ASR_DIALECT_SCHEMA[i];
        }
    }
    return NULL;
}

ASR_DialectOpKind ASR_DialectLookupSchemaByName(const char *mlir_name) {
    if (!mlir_name) {
        return ASR_DIALECT_OP_INVALID;
    }
    size_t lo = 0;
    size_t hi = ASR_DIALECT_SCHEMA_NAME_COUNT;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int cmp = strcmp(ASR_DIALECT_SCHEMA_BY_NAME[mid].mlir_name, mlir_name);
        if (cmp == 0) {
            return ASR_DIALECT_SCHEMA_BY_NAME[mid].kind;
        }
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return ASR_DIALECT_OP_INVALID;
}

void ASR_DialectClearVerifyError(void) {
    ASR_DialectClearVerifyErrorNative();
}

const ASR_DialectVerifyError *ASR_DialectGetLastVerifyError(void) {
    return ASR_DialectGetLastVerifyErrorNative();
}

void ASR_DialectClearCodegenError(void) {
    ASR_DialectClearCodegenErrorNative();
}

const ASR_DialectCodegenError *ASR_DialectGetLastCodegenError(void) {
    return ASR_DialectGetLastCodegenErrorNative();
}

bool ASR_DialectVerify(MLIR_Context *ctx, MLIR_OpHandle module) {
    // Future: ASR_DialectVerifyUpstream from asr_dialect_api_upstream.cpp
    return ASR_DialectVerifyNative(ctx, module);
}

bool ASR_DialectLowerToHighMLIR(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options) {
    // Future: ASR_DialectLowerToHighMLIRUpstream from asr_dialect_api_upstream.cpp
    return ASR_DialectLowerToHighMLIRNative(ctx, module, options);
}

string ASR_DialectPrint(MLIR_Context *ctx, MLIR_OpHandle module) {
    // Future: ASR_DialectPrintUpstream from asr_dialect_api_upstream.cpp
    return ASR_DialectPrintNative(ctx, module);
}

extern bool ASR_LowerUnsupportedNative(
    ASR_LoweringContext *ctx, MLIR_OpHandle op, const char *message);

bool ASR_LowerUnsupported(
    ASR_LoweringContext *ctx, MLIR_OpHandle op, const char *message) {
    return ASR_LowerUnsupportedNative(ctx, op, message);
}
