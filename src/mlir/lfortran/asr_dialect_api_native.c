// Native ASR dialect op storage, print, verify, and lowering driver.
#include "asr_dialect_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASR_KIND_ATTR str_lit("asr.op_kind")
#define ASR_FIELD_PREFIX "asr.f."

static string asr_field_attr_name(Arena *arena, const char *field_name) {
    size_t n = strlen(ASR_FIELD_PREFIX) + strlen(field_name);
    char *buf = (char *)arena_alloc(arena, n + 1);
    memcpy(buf, ASR_FIELD_PREFIX, strlen(ASR_FIELD_PREFIX));
    memcpy(buf + strlen(ASR_FIELD_PREFIX), field_name, strlen(field_name));
    buf[n] = '\0';
    return str_from_cstr(buf);
}

static MLIR_AttributeHandle field_to_attr(
    MLIR_Context *ctx, Arena *arena, const ASR_DialectField *field) {
    string aname = asr_field_attr_name(arena, field->name);
    switch (field->kind) {
        case ASR_FIELD_I64:
        case ASR_FIELD_I64_OPT:
            return MLIR_CreateAttributeInteger(ctx, aname, field->value.i64,
                MLIR_CreateTypeInteger(ctx, 64, false));
        case ASR_FIELD_F64:
            return MLIR_CreateAttributeFloat(ctx, aname, field->value.f64,
                MLIR_CreateTypeFloat(ctx, 64, false));
        case ASR_FIELD_BOOL:
        case ASR_FIELD_BOOL_OPT:
            return MLIR_CreateAttributeBool(ctx, aname, field->value.b);
        case ASR_FIELD_STRING:
        case ASR_FIELD_IDENTIFIER:
        case ASR_FIELD_STRING_OPT:
        case ASR_FIELD_IDENTIFIER_OPT:
            return MLIR_CreateAttributeString(ctx, aname, field->value.str);
        case ASR_FIELD_TTYPE:
        case ASR_FIELD_TTYPE_OPT:
            return MLIR_CreateAttributeType(ctx, aname, field->value.type);
        case ASR_FIELD_EXPR:
        case ASR_FIELD_EXPR_OPT:
        case ASR_FIELD_STMT:
        case ASR_FIELD_STMT_OPT:
        case ASR_FIELD_OP:
            return MLIR_CreateAttributeInteger(ctx, aname, (int64_t)field->value.op,
                MLIR_CreateTypeInteger(ctx, 64, false));
        default:
            return MLIR_CreateAttributeInteger(ctx, aname,
                (int64_t)field->value.i64, MLIR_CreateTypeInteger(ctx, 64, false));
    }
}

MLIR_OpHandle ASR_DialectCreateOpNative(
    MLIR_Context *ctx, ASR_DialectOpKind kind, MLIR_LocationHandle loc,
    const ASR_DialectField *fields, size_t n_fields) {
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        return MLIR_INVALID_HANDLE;
    }

    Arena *arena = ctx->arena;
    size_t n_attrs = n_fields + 1;
    MLIR_AttributeHandle *attrs =
        (MLIR_AttributeHandle *)arena_alloc(arena, n_attrs * sizeof(MLIR_AttributeHandle));
    attrs[0] = MLIR_CreateAttributeInteger(ctx, ASR_KIND_ATTR, (int64_t)kind,
        MLIR_CreateTypeInteger(ctx, 64, false));
    for (size_t i = 0; i < n_fields; ++i) {
        attrs[i + 1] = field_to_attr(ctx, arena, &fields[i]);
    }

    MLIR_TypeHandle result_ty = MLIR_INVALID_HANDLE;
    if (schema->category == ASR_DIALECT_CATEGORY_EXPR ||
        schema->category == ASR_DIALECT_CATEGORY_TTYPE) {
        for (size_t i = 0; i < n_fields; ++i) {
            if (fields[i].kind == ASR_FIELD_TTYPE ||
                fields[i].kind == ASR_FIELD_TTYPE_OPT) {
                result_ty = fields[i].value.type;
                break;
            }
        }
    }

    MLIR_ValueHandle result = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle rts[1] = {result_ty};
    MLIR_ValueHandle rs[1] = {MLIR_INVALID_HANDLE};
    if (result_ty != MLIR_INVALID_HANDLE) {
        char name_buf[32];
        snprintf(name_buf, sizeof(name_buf), "asr_v%zu", (size_t)kind);
        result = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, result_ty,
            str_from_cstr(name_buf), loc);
        rs[0] = result;
    }

    string opname = str_from_cstr(schema->mlir_name);
    return MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, opname, attrs, n_attrs,
        result_ty != MLIR_INVALID_HANDLE ? rts : NULL,
        result_ty != MLIR_INVALID_HANDLE ? 1 : 0,
        result_ty != MLIR_INVALID_HANDLE ? rs : NULL,
        result_ty != MLIR_INVALID_HANDLE ? 1 : 0,
        NULL, 0, NULL, 0, loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
}

ASR_DialectOpKind ASR_DialectGetOpKindNative(MLIR_OpHandle op) {
    MLIR_AttributeHandle kind_attr =
        MLIR_GetOpAttributeByName(op, "asr.op_kind");
    if (kind_attr == MLIR_INVALID_HANDLE) {
        return ASR_DIALECT_OP_INVALID;
    }
    return (ASR_DialectOpKind)MLIR_GetAttributeInteger(kind_attr);
}

static bool verify_one_op(MLIR_Context *ctx, MLIR_OpHandle op) {
    ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(op);
    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (!schema) {
        return false;
    }
    for (size_t i = 0; i < schema->n_fields; ++i) {
        if (schema->fields[i].presence == ASR_FIELD_REQUIRED) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s%s", ASR_FIELD_PREFIX, schema->fields[i].name);
            if (MLIR_GetOpAttributeByName(op, buf) == MLIR_INVALID_HANDLE) {
                (void)ctx;
                return false;
            }
        }
    }
    return true;
}

bool ASR_DialectVerifyNative(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (MLIR_GetOpType(module) != OP_TYPE_MODULE) {
        return false;
    }
    MLIR_RegionHandle region = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle block = MLIR_GetRegionBlock(region, 0);
    size_t n_ops = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < n_ops; ++i) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        ASR_DialectOpKind kind = ASR_DialectGetOpKindNative(op);
        if (kind == ASR_DIALECT_OP_INVALID) {
            continue;
        }
        if (!verify_one_op(ctx, op)) {
            return false;
        }
    }
    return true;
}

string ASR_DialectPrintNative(MLIR_Context *ctx, MLIR_OpHandle module) {
    (void)ctx;
    return MLIR_PrintOperationGeneric(ctx, module);
}

// Lowering context and handlers live in asr_dialect_lowering_handlers.c
extern bool ASR_DialectLowerModuleNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options);

bool ASR_DialectLowerToHighMLIRNative(
    MLIR_Context *ctx, MLIR_OpHandle module, const ASR_DialectOptions *options) {
    if (options && options->verify_asr_dialect) {
        if (!ASR_DialectVerifyNative(ctx, module)) {
            return false;
        }
    }
    return ASR_DialectLowerModuleNative(ctx, module, options);
}
