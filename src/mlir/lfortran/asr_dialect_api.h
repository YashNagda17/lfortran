// Public ASR MLIR dialect API — stable compiler-facing contract over mlir_api.h.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include <mlir_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ASR_DIALECT_BACKEND_NATIVE = 0,
    ASR_DIALECT_BACKEND_UPSTREAM = 1
} ASR_DialectBackend;

typedef struct {
    ASR_DialectBackend backend;
    bool verify_asr_dialect;
    bool allow_unimplemented_nodes;
    bool keep_asr_snapshot;
} ASR_DialectOptions;

#include "generated/asr_dialect_schema.inc"

typedef struct {
    ASR_DialectFieldKind kind;
    const char *name;
    union {
        int64_t i64;
        double f64;
        bool b;
        string str;
        MLIR_ValueHandle value;
        MLIR_TypeHandle type;
        MLIR_AttributeHandle attr;
        MLIR_RegionHandle region;
        MLIR_OpHandle op;
    } value;
} ASR_DialectField;

typedef struct ASR_LoweringContext ASR_LoweringContext;

MLIR_OpHandle ASR_DialectCreateOp(
    MLIR_Context *ctx,
    ASR_DialectOpKind kind,
    MLIR_LocationHandle loc,
    const ASR_DialectField *fields,
    size_t n_fields);

ASR_DialectOpKind ASR_DialectGetOpKind(MLIR_OpHandle op);

const ASR_DialectOpSchema *ASR_DialectLookupSchema(ASR_DialectOpKind kind);

bool ASR_DialectVerify(MLIR_Context *ctx, MLIR_OpHandle asr_module);

bool ASR_DialectLowerToHighMLIR(
    MLIR_Context *ctx,
    MLIR_OpHandle asr_module,
    const ASR_DialectOptions *options);

// Default: hybrid readable dump (alternative_2.md). Set LFORTRAN_ASR_DIALECT_DUMP=generic
// for raw MLIR with asr.f.* handle attributes (debug only).
string ASR_DialectPrint(MLIR_Context *ctx, MLIR_OpHandle module);

bool ASR_LowerUnsupported(
    ASR_LoweringContext *ctx,
    MLIR_OpHandle op,
    const char *message);

#ifdef __cplusplus
}
#endif
