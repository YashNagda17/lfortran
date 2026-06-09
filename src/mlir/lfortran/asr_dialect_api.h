// Public ASR MLIR dialect API — stable compiler-facing contract over mlir_api.h.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include <mlir_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Symbol reference in ASR dialect IR (stored as interned string metadata). */
typedef string ASR_SymbolRef;

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

#include "generated/asr_dialect_schema.h"

#include "generated/asr_dialect_api_generated.h"

typedef struct {
    MLIR_OpHandle *items;
    size_t n_items;
} ASR_DialectOpSeq;

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
        ASR_DialectOpSeq op_seq;
    } value;
} ASR_DialectField;

#define ASR_MAX_SYMS 256

typedef struct {
    string name;
    MLIR_OpHandle decl_op;  /* declaring asr.variable (like ASR::Variable_t *). */
    MLIR_ValueHandle memref;
    MLIR_TypeHandle memref_ty;
    bool is_array;
    int64_t array_len;
} ASR_SymSlot;

typedef struct ASR_LoweringContext {
    MLIR_Context *ctx;
    MLIR_LocationHandle loc;
    MLIR_OpHandle module;
    MLIR_BlockHandle module_block;
    MLIR_RegionHandle fn_region;
    MLIR_BlockHandle cur_block;
    MLIR_TypeHandle i32_ty;
    MLIR_TypeHandle i64_ty;
    MLIR_TypeHandle i1_ty;
    MLIR_TypeHandle index_ty;
    MLIR_TypeHandle memref_i32_1_ty;
    int ssa_counter;
    bool block_terminated;
    const ASR_DialectOptions *options;
    ASR_SymSlot sym_slots[ASR_MAX_SYMS];
    size_t n_sym_slots;
    MLIR_OpHandle site_parent;
    const char *site_region_field;
    size_t site_stmt_index;
    MLIR_OpHandle site_stmt_op;
} ASR_LoweringContext;

void ASR_AttachFortranLocAttrs(MLIR_Context *ctx, MLIR_OpHandle op,
        uint32_t loc_first, uint32_t loc_last);
bool ASR_ReadFortranLocAttrs(MLIR_OpHandle op,
        uint32_t *loc_first, uint32_t *loc_last);

MLIR_OpHandle ASR_DialectCreateOp(
    MLIR_Context *ctx,
    ASR_DialectOpKind kind,
    MLIR_LocationHandle loc,
    const ASR_DialectField *fields,
    size_t n_fields);

ASR_DialectOpKind ASR_DialectGetOpKind(MLIR_OpHandle op);

const ASR_DialectOpSchema *ASR_DialectLookupSchema(ASR_DialectOpKind kind);

ASR_DialectOpKind ASR_DialectLookupSchemaByName(const char *mlir_name);

typedef struct {
    char message[512];
    char filename[256];
    int line;
    int column;
    char op_name[128];
    char region_path[256];
    char stmt_detail[256];
    size_t dialect_line;
    uint32_t fortran_loc_first;
    uint32_t fortran_loc_last;
} ASR_DialectVerifyError;

void ASR_DialectClearVerifyError(void);
const ASR_DialectVerifyError *ASR_DialectGetLastVerifyError(void);

typedef struct {
    char message[512];
    char filename[256];
    int line;
    int column;
    char op_name[128];
    char region_path[256];
    char stmt_detail[256];
    size_t dialect_line;
    char expr_detail[128];
    uint32_t fortran_loc_first;
    uint32_t fortran_loc_last;
} ASR_DialectCodegenError;

void ASR_DialectClearCodegenError(void);
const ASR_DialectCodegenError *ASR_DialectGetLastCodegenError(void);

bool ASR_DialectVerify(MLIR_Context *ctx, MLIR_OpHandle asr_module);

bool ASR_DialectLowerToHighMLIR(
    MLIR_Context *ctx,
    MLIR_OpHandle asr_module,
    const ASR_DialectOptions *options);

string ASR_DialectPrint(MLIR_Context *ctx, MLIR_OpHandle module);

bool ASR_LowerUnsupported(
    ASR_LoweringContext *ctx,
    MLIR_OpHandle op,
    const char *message);

#ifdef __cplusplus
}
#endif
