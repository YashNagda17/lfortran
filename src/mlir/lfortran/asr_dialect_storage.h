// Native ASR dialect field readers (operands, attributes, regions).
#pragma once

#include <mlir_api.h>
#include "generated/asr_dialect_schema.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

ASR_DialectOpKind ASR_DialectLookupSchemaByName(const char *mlir_name);
const ASR_DialectOpSchema *ASR_DialectLookupSchema(ASR_DialectOpKind kind);

#ifdef __cplusplus
}
#endif

#define ASR_DIALECT_META_PREFIX "asr."
#define ASR_MAX_VARIADIC_FIELD_OPS 512

#if defined(__cplusplus)
#define ASR_THREAD_LOCAL thread_local
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define ASR_THREAD_LOCAL _Thread_local
#else
#define ASR_THREAD_LOCAL __thread
#endif

static inline MLIR_AttributeHandle asr_get_meta_attr(
        MLIR_OpHandle op, const char *field) {
    if (op == MLIR_INVALID_HANDLE || !field) {
        return MLIR_INVALID_HANDLE;
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "%s%s", ASR_DIALECT_META_PREFIX, field);
    return MLIR_GetOpAttributeByName(op, buf);
}

static inline MLIR_AttributeHandle asr_get_field_attr(
        MLIR_OpHandle op, const char *field) {
    return asr_get_meta_attr(op, field);
}

static inline ASR_DialectOpKind asr_op_kind_from_handle(MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return ASR_DIALECT_OP_INVALID;
    }
    string name = MLIR_GetOpName(op);
    if (!name.str || name.size == 0) {
        return ASR_DIALECT_OP_INVALID;
    }
    char buf[128];
    size_t n = name.size < sizeof(buf) - 1 ? name.size : sizeof(buf) - 1;
    memcpy(buf, name.str, n);
    buf[n] = '\0';
    return ASR_DialectLookupSchemaByName(buf);
}

static inline const ASR_DialectFieldDesc *asr_lookup_field_desc(
        MLIR_OpHandle op, const char *field) {
    const ASR_DialectOpSchema *schema =
        ASR_DialectLookupSchema(asr_op_kind_from_handle(op));
    if (!schema) {
        return NULL;
    }
    for (size_t i = 0; i < schema->n_fields; ++i) {
        if (strcmp(schema->fields[i].name, field) == 0) {
            return &schema->fields[i];
        }
    }
    return NULL;
}

static inline MLIR_OpHandle asr_get_field_op(MLIR_OpHandle op, const char *field) {
    if (op == MLIR_INVALID_HANDLE || !field) {
        return MLIR_INVALID_HANDLE;
    }
    const ASR_DialectFieldDesc *fd = asr_lookup_field_desc(op, field);
    if (fd) {
        if (fd->storage == ASR_STORAGE_OPERAND ||
                fd->storage == ASR_STORAGE_OPTIONAL_OPERAND) {
            if (fd->operand_index >= 0 &&
                    (size_t)fd->operand_index < MLIR_GetOpNumOperands(op)) {
                MLIR_ValueHandle v =
                    MLIR_GetOpOperand(op, (size_t)fd->operand_index);
                if (v != MLIR_INVALID_HANDLE) {
                    return MLIR_GetValueDefiningOp(v);
                }
            }
        } else if (fd->storage == ASR_STORAGE_VARIADIC_OPERANDS &&
                MLIR_GetOpNumOperands(op) > 0) {
            size_t start = fd->operand_index >= 0
                ? (size_t)fd->operand_index : 0;
            if (start < MLIR_GetOpNumOperands(op)) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, start);
                if (v != MLIR_INVALID_HANDLE) {
                    return MLIR_GetValueDefiningOp(v);
                }
            }
        }
    }
    return MLIR_INVALID_HANDLE;
}

static inline int64_t asr_get_field_i64(
        MLIR_OpHandle op, const char *field, int64_t def) {
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return def;
    }
    return MLIR_GetAttributeInteger(a);
}

static inline bool asr_get_field_bool(
        MLIR_OpHandle op, const char *field, bool def) {
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return def;
    }
    return MLIR_GetAttributeBool(a);
}

static inline string asr_get_field_str(MLIR_OpHandle op, const char *field) {
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return str_lit("");
    }
    return MLIR_GetAttributeString(a);
}

static inline MLIR_TypeHandle asr_get_field_type(MLIR_OpHandle op, const char *field) {
    if (op == MLIR_INVALID_HANDLE || !field) {
        return MLIR_INVALID_HANDLE;
    }
    if (strcmp(field, "type") == 0 && MLIR_GetOpNumResults(op) > 0) {
        return MLIR_GetValueType(MLIR_GetOpResult(op, 0));
    }
    MLIR_AttributeHandle a = asr_get_meta_attr(op, field);
    if (a == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    return MLIR_GetAttributeTypeValue(a);
}

static inline bool asr_field_attr_present(MLIR_OpHandle op, const char *field) {
    return asr_get_meta_attr(op, field) != MLIR_INVALID_HANDLE;
}

static inline size_t asr_region_block_op_count(MLIR_RegionHandle region) {
    if (region == MLIR_INVALID_HANDLE || MLIR_GetRegionNumBlocks(region) == 0) {
        return 0;
    }
    return MLIR_GetBlockNumOps(MLIR_GetRegionBlock(region, 0));
}

static inline MLIR_OpHandle asr_region_block_op(MLIR_RegionHandle region, size_t i) {
    if (region == MLIR_INVALID_HANDLE || MLIR_GetRegionNumBlocks(region) == 0) {
        return MLIR_INVALID_HANDLE;
    }
    MLIR_BlockHandle block = MLIR_GetRegionBlock(region, 0);
    if (i >= MLIR_GetBlockNumOps(block)) {
        return MLIR_INVALID_HANDLE;
    }
    return MLIR_GetBlockOp(block, i);
}

static inline MLIR_RegionHandle asr_get_field_region(
        MLIR_OpHandle op, const char *field) {
    const ASR_DialectFieldDesc *fd = asr_lookup_field_desc(op, field);
    if (!fd || (fd->storage != ASR_STORAGE_REGION &&
            fd->storage != ASR_STORAGE_OPTIONAL_REGION)) {
        return MLIR_INVALID_HANDLE;
    }
    if (fd->region_index < 0 ||
            (size_t)fd->region_index >= MLIR_GetOpNumRegions(op)) {
        return MLIR_INVALID_HANDLE;
    }
    return MLIR_GetOpRegion(op, (size_t)fd->region_index);
}

static ASR_THREAD_LOCAL MLIR_OpHandle asr_field_region_ops[ASR_MAX_VARIADIC_FIELD_OPS];
static ASR_THREAD_LOCAL size_t asr_field_region_ops_n = 0;

static inline MLIR_OpHandle *asr_get_field_op_seq(
        MLIR_OpHandle op, const char *field) {
    asr_field_region_ops_n = 0;
    const ASR_DialectFieldDesc *fd = asr_lookup_field_desc(op, field);
    if (!fd || (fd->storage != ASR_STORAGE_REGION &&
            fd->storage != ASR_STORAGE_OPTIONAL_REGION)) {
        return NULL;
    }
    MLIR_RegionHandle region = asr_get_field_region(op, field);
    size_t n = asr_region_block_op_count(region);
    for (size_t i = 0; i < n; ++i) {
        if (asr_field_region_ops_n >= ASR_MAX_VARIADIC_FIELD_OPS) {
            break;
        }
        MLIR_OpHandle child = asr_region_block_op(region, i);
        if (child != MLIR_INVALID_HANDLE) {
            asr_field_region_ops[asr_field_region_ops_n++] = child;
        }
    }
    return asr_field_region_ops_n > 0 ? asr_field_region_ops : NULL;
}

static inline size_t asr_get_field_op_seq_count(MLIR_OpHandle op, const char *field) {
    const ASR_DialectFieldDesc *fd = asr_lookup_field_desc(op, field);
    if (!fd) {
        return 0;
    }
    if (fd->storage == ASR_STORAGE_REGION ||
            fd->storage == ASR_STORAGE_OPTIONAL_REGION) {
        return asr_region_block_op_count(asr_get_field_region(op, field));
    }
    return 0;
}

static inline int64_t asr_memref_static_len(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (!MLIR_IsTypeMemref(ty)) {
        return 0;
    }
    string ts = MLIR_GetTypeString(ctx, ty);
    const char *prefix = "memref<";
    size_t plen = 7;
    if (ts.size < plen + 5 || memcmp(ts.str, prefix, plen) != 0) {
        return 0;
    }
    size_t i = plen;
    int64_t n = 0;
    while (i < ts.size && ts.str[i] >= '0' && ts.str[i] <= '9') {
        n = n * 10 + (int64_t)(ts.str[i] - '0');
        i++;
    }
    if (i + 4 >= ts.size || ts.str[i] != 'x') {
        return 0;
    }
    if (memcmp(ts.str + i + 1, "i32", 3) != 0 || ts.str[i + 4] != '>') {
        return 0;
    }
    return n > 0 ? n : 0;
}

static inline MLIR_TypeHandle asr_get_var_result_type(MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE || MLIR_GetOpNumResults(op) == 0) {
        return MLIR_INVALID_HANDLE;
    }
    return MLIR_GetValueType(MLIR_GetOpResult(op, 0));
}

static inline int64_t asr_get_var_array_len(MLIR_Context *ctx, MLIR_OpHandle var_op) {
    return asr_memref_static_len(ctx, asr_get_var_result_type(var_op));
}

static inline bool asr_var_is_array(MLIR_Context *ctx, MLIR_OpHandle var_op) {
    return asr_get_var_array_len(ctx, var_op) > 0;
}

static inline size_t asr_get_seq_n_args_attr(MLIR_OpHandle op) {
    size_t n = asr_get_field_op_seq_count(op, "args");
    if (n > 0) {
        return n;
    }
    n = asr_get_field_op_seq_count(op, "elements");
    if (n > 0) {
        return n;
    }
    return (size_t)asr_get_field_i64(op, "n_args", 0);
}

#define asr_get_field_expr_op asr_get_field_op
