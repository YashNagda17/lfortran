// ASR dialect API dispatch — routes to native or upstream backend.
#include "asr_dialect_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ASR_DialectBackend asr_dialect_active_backend(void) {
    const char *e = getenv("USE_MLIR_Upstream");
    if (e && e[0] == '1' && e[1] == '\0') {
        return ASR_DIALECT_BACKEND_UPSTREAM;
    }
    return ASR_DIALECT_BACKEND_NATIVE;
}

// Native backend (always linked).
extern MLIR_OpHandle ASR_DialectCreateOpNative(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields);
extern ASR_DialectOpKind ASR_DialectGetOpKindNative(MLIR_OpHandle op);
extern bool ASR_DialectVerifyNative(MLIR_Context *ctx, MLIR_OpHandle module);
extern bool ASR_DialectLowerToHighMLIRNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options);
extern string ASR_DialectPrintNative(MLIR_Context *ctx, MLIR_OpHandle module);

// Upstream backend (optional stub until full ODS registration).
extern MLIR_OpHandle ASR_DialectCreateOpUpstream(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields);
extern ASR_DialectOpKind ASR_DialectGetOpKindUpstream(MLIR_OpHandle op);
extern bool ASR_DialectVerifyUpstream(MLIR_Context *ctx, MLIR_OpHandle module);
extern bool ASR_DialectLowerToHighMLIRUpstream(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options);
extern string ASR_DialectPrintUpstream(MLIR_Context *ctx, MLIR_OpHandle module);

MLIR_OpHandle ASR_DialectCreateOp(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields) {
    if (asr_dialect_active_backend() == ASR_DIALECT_BACKEND_UPSTREAM) {
        return ASR_DialectCreateOpUpstream(ctx, kind, loc, fields, n_fields);
    }
    return ASR_DialectCreateOpNative(ctx, kind, loc, fields, n_fields);
}

ASR_DialectOpKind ASR_DialectGetOpKind(MLIR_OpHandle op) {
    if (asr_dialect_active_backend() == ASR_DIALECT_BACKEND_UPSTREAM) {
        return ASR_DialectGetOpKindUpstream(op);
    }
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

bool ASR_DialectVerify(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (asr_dialect_active_backend() == ASR_DIALECT_BACKEND_UPSTREAM) {
        return ASR_DialectVerifyUpstream(ctx, module);
    }
    return ASR_DialectVerifyNative(ctx, module);
}

bool ASR_DialectLowerToHighMLIR(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options) {
    if (asr_dialect_active_backend() == ASR_DIALECT_BACKEND_UPSTREAM) {
        return ASR_DialectLowerToHighMLIRUpstream(ctx, module, options);
    }
    return ASR_DialectLowerToHighMLIRNative(ctx, module, options);
}

string ASR_DialectPrint(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (asr_dialect_active_backend() == ASR_DIALECT_BACKEND_UPSTREAM) {
        return ASR_DialectPrintUpstream(ctx, module);
    }
    return ASR_DialectPrintNative(ctx, module);
}

bool ASR_LowerUnsupported(
    ASR_LoweringContext *ctx, MLIR_OpHandle op, const char *message) {
    (void)ctx;
    (void)op;
    if (message) {
        fprintf(stderr, "ASR dialect lowering: %s\n", message);
    }
    return false;
}
