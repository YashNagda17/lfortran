// Legacy ASR dialect text dump — formatting/layout only.
//
// Reads stored MLIR IR through generated accessors (same fields lowering uses).
// Semantic access for codegen lives in asr_dialect_lowering_handlers.c.
// Type spellings below are display-only; lowering must use generated fields.
#include "asr_dialect_api.h"
#include "asr_dialect_storage.h"
#include "generated/asr_dialect_print_helpers.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include <base/strbuf.h>
#include <base/string.h>

#define ASR_PP_INDENT 2
#define ASR_PP_UNRESOLVED "<unresolved expr>"

/* Display-only helpers: format stored MLIR types for legacy dialect text. */
static void pp_display_sym_ref(string name, char *buf, size_t bufsz) {
    if (!name.str || name.size == 0) {
        snprintf(buf, bufsz, "@?");
        return;
    }
    snprintf(buf, bufsz, "@%.*s", (int)name.size, name.str);
}

static int64_t pp_display_asr_kind_from_integer_type(
        MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string ts = MLIR_GetTypeString(ctx, ty);
    if (ts.size >= 3 && ts.str[0] == 'i') {
        int bits = 0;
        for (size_t i = 1; i < ts.size && ts.str[i] >= '0' && ts.str[i] <= '9'; ++i) {
            bits = bits * 10 + (ts.str[i] - '0');
        }
        switch (bits) {
            case 8: return 1;
            case 16: return 2;
            case 32: return 4;
            case 64: return 8;
            default: return 4;
        }
    }
    return 4;
}

static int64_t pp_display_asr_kind_from_float_type(
        MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string ts = MLIR_GetTypeString(ctx, ty);
    if (ts.size >= 3 && ts.str[0] == 'f') {
        int bits = 0;
        for (size_t i = 1; i < ts.size && ts.str[i] >= '0' && ts.str[i] <= '9'; ++i) {
            bits = bits * 10 + (ts.str[i] - '0');
        }
        switch (bits) {
            case 32: return 4;
            case 64: return 8;
            default: return 8;
        }
    }
    return 8;
}

static bool pp_display_mlir_type_spelling(
        MLIR_Context *ctx, MLIR_TypeHandle ty, char *buf, size_t bufsz) {
    if (ty == MLIR_INVALID_HANDLE) {
        return false;
    }
    if (MLIR_IsTypeInteger(ty)) {
        snprintf(buf, bufsz, "!asr.integer<%lld>",
            (long long)pp_display_asr_kind_from_integer_type(ctx, ty));
        return true;
    }
    if (MLIR_IsTypeFloat(ty)) {
        snprintf(buf, bufsz, "!asr.real<%lld>",
            (long long)pp_display_asr_kind_from_float_type(ctx, ty));
        return true;
    }
    if (MLIR_IsTypeMemref(ty)) {
        MLIR_TypeHandle elem = MLIR_GetTypeShapedElement(ty);
        char elem_ty[64];
        int64_t n = asr_memref_static_len(ctx, ty);
        if (n > 0 && pp_display_mlir_type_spelling(ctx, elem, elem_ty, sizeof(elem_ty))) {
            snprintf(buf, bufsz, "!asr.array<%s, %lld>", elem_ty, (long long)n);
            return true;
        }
    }
    return false;
}

static MLIR_TypeHandle pp_display_expr_result_type(MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    MLIR_TypeHandle ty = asr_get_field_type(op, "type");
    if (ty != MLIR_INVALID_HANDLE) {
        return ty;
    }
    if (MLIR_GetOpNumResults(op) > 0) {
        return MLIR_GetValueType(MLIR_GetOpResult(op, 0));
    }
    return MLIR_INVALID_HANDLE;
}

static bool pp_display_op_type_spelling(
        MLIR_Context *ctx, MLIR_OpHandle op, char *buf, size_t bufsz) {
    if (op == MLIR_INVALID_HANDLE) {
        return false;
    }
    if (pp_display_mlir_type_spelling(ctx, asr_get_field_type(op, "type"), buf, bufsz)) {
        return true;
    }
    return pp_display_mlir_type_spelling(
        ctx, pp_display_expr_result_type(op), buf, bufsz);
}

typedef struct {
    MLIR_Context *ctx;
    strbuf out;
    int indent;
} ASR_PpCtx;

static void pp_expr(ASR_PpCtx *pp, MLIR_OpHandle op, char *buf, size_t bufsz);
static void pp_format_index_op(ASR_PpCtx *pp, MLIR_OpHandle idx_op,
        char *buf, size_t bufsz);
static void pp_format_array_item(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz);
static void pp_format_print_arg(ASR_PpCtx *pp, MLIR_OpHandle text,
        char *buf, size_t bufsz);

static void pp_indent(ASR_PpCtx *pp) {
    for (int i = 0; i < pp->indent; ++i) {
        strbuf_append_char(pp->ctx->arena, &pp->out, ' ');
    }
}

static void pp_line(ASR_PpCtx *pp, const char *s) {
    pp_indent(pp);
    strbuf_append_cstr(pp->ctx->arena, &pp->out, s);
    strbuf_append_char(pp->ctx->arena, &pp->out, '\n');
}

static void pp_fmt(ASR_PpCtx *pp, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    pp_line(pp, buf);
}

static void pp_format_sym(string name, char *buf, size_t bufsz) {
    pp_display_sym_ref(name, buf, bufsz);
}

static MLIR_OpHandle pp_peel_cast(MLIR_OpHandle op) {
    for (int depth = 0; depth < 64; ++depth) {
        if (op == MLIR_INVALID_HANDLE ||
                ASR_DialectGetOpKind(op) != ASR_DIALECT_OP_EXPR_CAST) {
            return op;
        }
        op = asr_get_field_op(op, "arg");
    }
    return op;
}

/* Index/base formatting without calling pp_expr (avoids mutual recursion). */
static void pp_format_simple_expr(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    if (op == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "?");
        return;
    }
    op = pp_peel_cast(op);
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    if (kind == ASR_DIALECT_OP_EXPR_INTEGERCONSTANT) {
        snprintf(buf, bufsz, "%" PRId64, asr_get_field_i64(op, "n", 0));
        return;
    }
    if (kind == ASR_DIALECT_OP_EXPR_VAR) {
        pp_format_sym(asr_get_field_str(op, "v"), buf, bufsz);
        return;
    }
    pp_expr(pp, op, buf, bufsz);
}

static void pp_format_index_op(ASR_PpCtx *pp, MLIR_OpHandle idx_op,
        char *buf, size_t bufsz) {
    if (idx_op == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "?");
        return;
    }
    if (ASR_DialectGetOpKind(idx_op) == ASR_DIALECT_OP_ARRAY_INDEX_ARRAY_INDEX) {
        MLIR_OpHandle right = asr_get_field_op(idx_op, "right");
        MLIR_OpHandle left = asr_get_field_op(idx_op, "left");
        if (right != MLIR_INVALID_HANDLE) {
            pp_format_simple_expr(pp, right, buf, bufsz);
            return;
        }
        if (left != MLIR_INVALID_HANDLE) {
            pp_format_simple_expr(pp, left, buf, bufsz);
            return;
        }
    }
    pp_format_simple_expr(pp, idx_op, buf, bufsz);
}

static bool pp_format_array_constant(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    MLIR_OpHandle *elems = asr_get_field_op_seq(op, "elements");
    size_t n = asr_get_field_op_seq_count(op, "elements");
    if (!elems || n == 0) {
        return false;
    }
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, bufsz - pos, "[");
    for (size_t i = 0; i < n && pos < bufsz - 16; ++i) {
        char ebuf[64];
        if (i > 0) {
            pos += (size_t)snprintf(buf + pos, bufsz - pos, ", ");
        }
        pp_expr(pp, elems[i], ebuf, sizeof(ebuf));
        pos += (size_t)snprintf(buf + pos, bufsz - pos, "%s", ebuf);
    }
    snprintf(buf + pos, bufsz - pos, "]");
    return true;
}

static bool pp_format_array_constructor(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    MLIR_OpHandle *args = asr_get_field_op_seq(op, "args");
    size_t n = asr_get_seq_n_args_attr(op);
    if (!args || n == 0) {
        return false;
    }
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, bufsz - pos, "[");
    for (size_t i = 0; i < n && pos < bufsz - 16; ++i) {
        char ebuf[64];
        if (i > 0) {
            pos += (size_t)snprintf(buf + pos, bufsz - pos, ", ");
        }
        pp_expr(pp, args[i], ebuf, sizeof(ebuf));
        pos += (size_t)snprintf(buf + pos, bufsz - pos, "%s", ebuf);
    }
    snprintf(buf + pos, bufsz - pos, "]");
    return true;
}

static bool pp_is_array_item_op(MLIR_OpHandle op) {
    return op != MLIR_INVALID_HANDLE &&
        ASR_DialectGetOpKind(op) == ASR_DIALECT_OP_EXPR_ARRAYITEM;
}

static void pp_format_array_item(ASR_PpCtx *pp, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    MLIR_OpHandle base = asr_get_field_op(op, "v");
    char basebuf[128];
    char ibuf[64];
    if (ASR_DialectGetOpKind(base) == ASR_DIALECT_OP_EXPR_VAR) {
        pp_format_sym(asr_get_field_str(base, "v"), basebuf, sizeof(basebuf));
    } else {
        pp_expr(pp, base, basebuf, sizeof(basebuf));
    }
    MLIR_OpHandle *indices = asr_get_field_op_seq(op, "args");
    size_t n_idx = asr_get_seq_n_args_attr(op);
    if (indices && n_idx > 0) {
        pp_format_index_op(pp, indices[0], ibuf, sizeof(ibuf));
        snprintf(buf, bufsz, "%s(%s)", basebuf, ibuf);
        return;
    }
    if (indices && indices[0] != MLIR_INVALID_HANDLE) {
        pp_format_index_op(pp, indices[0], ibuf, sizeof(ibuf));
        snprintf(buf, bufsz, "%s(%s)", basebuf, ibuf);
        return;
    }
    snprintf(buf, bufsz, "%s(?)", basebuf);
}

static void pp_format_print_arg(ASR_PpCtx *pp, MLIR_OpHandle text,
        char *buf, size_t bufsz) {
    if (text == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "%s", ASR_PP_UNRESOLVED);
        return;
    }
    if (ASR_DialectGetOpKind(text) == ASR_DIALECT_OP_EXPR_VAR) {
        pp_format_sym(asr_get_field_str(text, "v"), buf, bufsz);
        return;
    }
    if (pp_is_array_item_op(text)) {
        pp_format_array_item(pp, text, buf, bufsz);
        return;
    }
    if (ASR_DialectGetOpKind(text) == ASR_DIALECT_OP_EXPR_STRINGFORMAT) {
        MLIR_OpHandle *args = asr_get_field_op_seq(text, "args");
        size_t n = asr_get_seq_n_args_attr(text);
        if (args && n > 0) {
            pp_format_print_arg(pp, args[0], buf, bufsz);
            return;
        }
    }
    pp_expr(pp, text, buf, bufsz);
}

static bool pp_binop_infix_spelling(int64_t bop, char *op, size_t opsz) {
    switch (bop) {
        case 0: snprintf(op, opsz, "+"); return true;
        case 1: snprintf(op, opsz, "-"); return true;
        case 2: snprintf(op, opsz, "*"); return true;
        case 3: snprintf(op, opsz, "/"); return true;
        default: return false;
    }
}

static void pp_expr(ASR_PpCtx *pp, MLIR_OpHandle op, char *buf, size_t bufsz) {
    if (op == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "%s", ASR_PP_UNRESOLVED);
        return;
    }

    op = pp_peel_cast(op);
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    switch (kind) {
        case ASR_DIALECT_OP_EXPR_INTEGERCONSTANT:
            snprintf(buf, bufsz, "%" PRId64, asr_get_field_i64(op, "n", 0));
            return;
        case ASR_DIALECT_OP_EXPR_VAR:
            pp_format_sym(ASR_VarV(op), buf, bufsz);
            return;
        case ASR_DIALECT_OP_EXPR_INTEGERBINOP: {
            MLIR_OpHandle left = asr_get_field_op(op, "left");
            MLIR_OpHandle right = asr_get_field_op(op, "right");
            int64_t bop = asr_get_field_i64(op, "op", 0);
            char lbuf[160];
            char rbuf[160];
            char opch[8];
            pp_expr(pp, left, lbuf, sizeof(lbuf));
            pp_expr(pp, right, rbuf, sizeof(rbuf));
            if (pp_binop_infix_spelling(bop, opch, sizeof(opch))) {
                snprintf(buf, bufsz, "%s %s %s", lbuf, opch, rbuf);
                return;
            }
            const char *opkw = asr_enum_binop_name(bop);
            if (!opkw) {
                opkw = "add";
            }
            snprintf(buf, bufsz, "asr.integer_bin_op %s(%s, %s)",
                opkw, lbuf, rbuf);
            return;
        }
        case ASR_DIALECT_OP_EXPR_ARRAYCONSTANT:
            if (pp_format_array_constant(pp, op, buf, bufsz)) {
                return;
            }
            snprintf(buf, bufsz, "[?]");
            return;
        case ASR_DIALECT_OP_EXPR_ARRAYCONSTRUCTOR:
            if (pp_format_array_constructor(pp, op, buf, bufsz)) {
                return;
            }
            snprintf(buf, bufsz, "[?]");
            return;
        case ASR_DIALECT_OP_EXPR_ARRAYITEM:
            pp_format_array_item(pp, op, buf, bufsz);
            return;
        default:
            break;
    }

    if (pp_is_array_item_op(op)) {
        pp_format_array_item(pp, op, buf, bufsz);
        return;
    }

    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (schema && schema->mlir_name) {
        snprintf(buf, bufsz, "%s", schema->mlir_name);
    } else {
        snprintf(buf, bufsz, "%s", ASR_PP_UNRESOLVED);
    }
}

static bool pp_is_default_step(MLIR_OpHandle step_op) {
    if (step_op == MLIR_INVALID_HANDLE) {
        return true;
    }
    if (ASR_DialectGetOpKind(step_op) == ASR_DIALECT_OP_EXPR_INTEGERCONSTANT) {
        return asr_get_field_i64(step_op, "n", 0) == 1;
    }
    return false;
}

static void pp_print_variable_full_attrs(ASR_PpCtx *pp, MLIR_OpHandle op) {
    const ASR_DialectOpSchema *schema =
        ASR_DialectLookupSchema(ASR_DIALECT_OP_SYMBOL_VARIABLE);
    if (!schema) {
        return;
    }

    pp->indent += ASR_PP_INDENT;

    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        const char *fname = fd->name;
        if (strcmp(fname, "name") == 0 || strcmp(fname, "type") == 0 ||
                strcmp(fname, "parent_symtab") == 0) {
            continue;
        }
        if (strcmp(fname, "dependencies") == 0) {
            string deps = asr_get_field_str(op, fname);
            if (deps.size == 0) {
                pp_fmt(pp, "dependencies = [],");
            } else {
                pp_fmt(pp, "dependencies = [%.*s],", (int)deps.size,
                    deps.str ? deps.str : "");
            }
            continue;
        }

        if (strcmp(fname, "codims") == 0) {
            size_t n_codims = asr_get_field_op_seq_count(op, fname);
            if (n_codims == 0) {
                pp_fmt(pp, "codims = [],");
            } else {
                pp_fmt(pp, "codims = [%zu],", n_codims);
            }
            continue;
        }

        if (fd->kind == ASR_FIELD_EXPR_OPT || fd->kind == ASR_FIELD_EXPR) {
            MLIR_OpHandle child = asr_get_field_op(op, fname);
            if (child == MLIR_INVALID_HANDLE) {
                pp_fmt(pp, "  %s = null,", fname);
            } else {
                char ebuf[160];
                pp_expr(pp, child, ebuf, sizeof(ebuf));
                pp_fmt(pp, "  %s = %s,", fname, ebuf);
            }
            continue;
        }

        char line[256];
        if (asr_pp_format_variable_scalar_field(op, fname, line, sizeof(line))) {
            pp_line(pp, line);
        }
    }

    pp->indent -= ASR_PP_INDENT;
    pp_line(pp, "}");
}

static void pp_stmt(ASR_PpCtx *pp, MLIR_OpHandle op);

static void pp_region_block_stmts(ASR_PpCtx *pp, MLIR_RegionHandle region) {
    size_t n = asr_region_block_op_count(region);
    for (size_t i = 0; i < n; ++i) {
        pp_stmt(pp, asr_region_block_op(region, i));
    }
}

static void pp_scope_regions(ASR_PpCtx *pp, MLIR_OpHandle scope_op) {
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(scope_op);
    MLIR_RegionHandle symtab_r = MLIR_INVALID_HANDLE;
    MLIR_RegionHandle metadata_r = MLIR_INVALID_HANDLE;
    MLIR_RegionHandle body_r = MLIR_INVALID_HANDLE;
    if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM) {
        symtab_r = ASR_ProgramGetSymtabRegion(scope_op);
        metadata_r = ASR_ProgramGetMetadataRegion(scope_op);
        body_r = ASR_ProgramGetBodyRegion(scope_op);
    } else if (kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
        symtab_r = ASR_FunctionGetSymtabRegion(scope_op);
        metadata_r = ASR_FunctionGetMetadataRegion(scope_op);
        body_r = ASR_FunctionGetBodyRegion(scope_op);
    }

    pp_line(pp, "asr.symtab {");
    pp->indent += ASR_PP_INDENT;
    pp_region_block_stmts(pp, symtab_r);
    pp->indent -= ASR_PP_INDENT;
    pp_line(pp, "}");

    pp_line(pp, "asr.metadata {");
    pp->indent += ASR_PP_INDENT;
    pp_region_block_stmts(pp, metadata_r);
    pp->indent -= ASR_PP_INDENT;
    pp_line(pp, "}");

    pp_line(pp, "asr.body {");
    pp->indent += ASR_PP_INDENT;
    pp_region_block_stmts(pp, body_r);
    pp->indent -= ASR_PP_INDENT;
    pp_line(pp, "}");
}

static void pp_stmt(ASR_PpCtx *pp, MLIR_OpHandle op) {
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    char buf[256];
    char ty[32];

    if (kind == ASR_DIALECT_OP_INVALID) {
        return;
    }

    if (kind == ASR_DIALECT_OP_UNIT_TRANSLATIONUNIT) {
        pp_line(pp, "asr.translation_unit {");
        pp->indent += ASR_PP_INDENT;
        pp_region_block_stmts(pp, ASR_TranslationUnitGetItemsRegion(op));
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        return;
    }

    if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, buf, sizeof(buf));
        pp_fmt(pp, "asr.program %s {", buf);
        pp->indent += ASR_PP_INDENT;
        pp_scope_regions(pp, op);
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        return;
    }

    if (kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, buf, sizeof(buf));
        pp_fmt(pp, "asr.function %s {", buf);
        pp->indent += ASR_PP_INDENT;
        pp_scope_regions(pp, op);
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        return;
    }

    if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
        string name = asr_get_field_str(op, "name");
        int64_t arr_len = asr_get_var_array_len(pp->ctx, op);
        pp_format_sym(name, buf, sizeof(buf));
        if (pp_display_op_type_spelling(pp->ctx, op, ty, sizeof(ty))) {
            pp_fmt(pp, "asr.variable %s : %s {", buf, ty);
        } else if (arr_len > 0) {
            pp_fmt(pp, "asr.variable %s : i32[%lld] {", buf, (long long)arr_len);
        } else {
            pp_fmt(pp, "asr.variable %s {", buf);
        }
        pp_print_variable_full_attrs(pp, op);
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_ASSIGNMENT) {
        MLIR_OpHandle target = ASR_AssignmentTargetOp(op);
        MLIR_OpHandle value = ASR_AssignmentValueOp(op);
        char tbuf[160];
        char vbuf[160];
        if (ASR_DialectGetOpKind(target) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(ASR_VarV(target), tbuf, sizeof(tbuf));
        } else {
            pp_expr(pp, target, tbuf, sizeof(tbuf));
        }
        pp_expr(pp, value, vbuf, sizeof(vbuf));
        if (pp_display_op_type_spelling(pp->ctx, target, ty, sizeof(ty))) {
            pp_fmt(pp, "asr.assign %s to %s : %s", vbuf, tbuf, ty);
        } else {
            pp_fmt(pp, "asr.assign %s to %s", vbuf, tbuf);
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_DOLOOP) {
        MLIR_RegionHandle head_r = ASR_DoLoopGetHeadRegion(op);
        MLIR_OpHandle head = asr_region_block_op(head_r, 0);
        MLIR_RegionHandle body_r = ASR_DoLoopGetBodyRegion(op);
        MLIR_OpHandle v = ASR_do_loop_headVOp(head);
        MLIR_OpHandle start = ASR_do_loop_headStartOp(head);
        MLIR_OpHandle end = ASR_do_loop_headEndOp(head);
        MLIR_OpHandle step = ASR_do_loop_headIncrementOp(head);
        char iv[64];
        char sbuf[160];
        char ebuf[160];
        char stepbuf[160];

        if (ASR_DialectGetOpKind(v) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(ASR_VarV(v), iv, sizeof(iv));
        } else {
            pp_expr(pp, v, iv, sizeof(iv));
        }
        pp_expr(pp, start, sbuf, sizeof(sbuf));
        pp_expr(pp, end, ebuf, sizeof(ebuf));

        if (!pp_is_default_step(step)) {
            pp_expr(pp, step, stepbuf, sizeof(stepbuf));
            pp_fmt(pp, "asr.do_loop %s = %s to %s step %s {",
                iv, sbuf, ebuf, stepbuf);
        } else {
            pp_fmt(pp, "asr.do_loop %s = %s to %s {", iv, sbuf, ebuf);
        }

        pp->indent += ASR_PP_INDENT;
        pp_region_block_stmts(pp, body_r);
        pp->indent -= ASR_PP_INDENT;

        MLIR_RegionHandle orelse_r = ASR_DoLoopGetOrelseRegion(op);
        if (asr_region_block_op_count(orelse_r) > 0) {
            pp_line(pp, "} else {");
            pp->indent += ASR_PP_INDENT;
            pp_region_block_stmts(pp, orelse_r);
            pp->indent -= ASR_PP_INDENT;
            pp_line(pp, "}");
        } else {
            pp_line(pp, "}");
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_IF) {
        MLIR_OpHandle test = ASR_IfTestOp(op);
        char cbuf[160];
        pp_expr(pp, test, cbuf, sizeof(cbuf));
        pp_fmt(pp, "asr.if %s {", cbuf);

        pp->indent += ASR_PP_INDENT;
        pp_region_block_stmts(pp, ASR_IfGetBodyRegion(op));
        pp->indent -= ASR_PP_INDENT;

        MLIR_RegionHandle orelse_r = ASR_IfGetOrelseRegion(op);
        if (asr_region_block_op_count(orelse_r) > 0) {
            pp_line(pp, "} else {");
            pp->indent += ASR_PP_INDENT;
            pp_region_block_stmts(pp, orelse_r);
            pp->indent -= ASR_PP_INDENT;
            pp_line(pp, "}");
        } else {
            pp_line(pp, "}");
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_WHILELOOP) {
        MLIR_OpHandle test = ASR_WhileLoopTestOp(op);
        char cbuf[160];
        pp_expr(pp, test, cbuf, sizeof(cbuf));
        pp_fmt(pp, "asr.while_loop %s {", cbuf);

        pp->indent += ASR_PP_INDENT;
        pp_region_block_stmts(pp, ASR_WhileLoopGetBodyRegion(op));
        pp->indent -= ASR_PP_INDENT;

        MLIR_RegionHandle orelse_r = ASR_WhileLoopGetOrelseRegion(op);
        if (asr_region_block_op_count(orelse_r) > 0) {
            pp_line(pp, "} else {");
            pp->indent += ASR_PP_INDENT;
            pp_region_block_stmts(pp, orelse_r);
            pp->indent -= ASR_PP_INDENT;
            pp_line(pp, "}");
        } else {
            pp_line(pp, "}");
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_PRINT) {
        MLIR_OpHandle text = ASR_PrintTextOp(op);
        char pbuf[160];
        pp_format_print_arg(pp, text, pbuf, sizeof(pbuf));
        pp_fmt(pp, "asr.print %s", pbuf);
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_RETURN) {
        pp_line(pp, "asr.return");
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_SUBROUTINECALL) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, buf, sizeof(buf));
        pp_fmt(pp, "asr.subroutine_call %s", buf);
        return;
    }

    /* Expression ops are not printed at module level — only inline in statements. */
    if (kind == ASR_DIALECT_OP_EXPR_INTEGERCONSTANT ||
            kind == ASR_DIALECT_OP_EXPR_INTEGERBINOP ||
            kind == ASR_DIALECT_OP_EXPR_VAR ||
            kind == ASR_DIALECT_OP_EXPR_ARRAYITEM ||
            kind == ASR_DIALECT_OP_EXPR_ARRAYCONSTANT ||
            kind == ASR_DIALECT_OP_EXPR_ARRAYCONSTRUCTOR) {
        return;
    }

    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (schema && schema->mlir_name) {
        pp_fmt(pp, "%s", schema->mlir_name);
    }
}

void ASR_DialectFormatOpSummary(MLIR_Context *ctx, MLIR_OpHandle op,
        char *buf, size_t bufsz) {
    if (!buf || bufsz == 0) {
        return;
    }
    buf[0] = '\0';
    if (op == MLIR_INVALID_HANDLE) {
        snprintf(buf, bufsz, "<invalid op handle>");
        return;
    }
    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    if (kind == ASR_DIALECT_OP_INVALID) {
        string raw = MLIR_GetOpName(op);
        if (raw.str && raw.size > 0) {
            size_t n = raw.size < bufsz - 16 ? raw.size : bufsz - 16;
            snprintf(buf, bufsz, "%.*s (unregistered)", (int)n, raw.str);
        } else {
            snprintf(buf, bufsz, "<unregistered op>");
        }
        return;
    }

    ASR_PpCtx pp = {};
    pp.ctx = ctx;
    char line[512];
    char ty[32];
    char tbuf[160];
    char vbuf[160];

    if (kind == ASR_DIALECT_OP_STMT_ASSIGNMENT) {
        MLIR_OpHandle target = ASR_AssignmentTargetOp(op);
        MLIR_OpHandle value = ASR_AssignmentValueOp(op);
        if (ASR_DialectGetOpKind(target) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(ASR_VarV(target), tbuf, sizeof(tbuf));
        } else {
            pp_format_simple_expr(&pp, target, tbuf, sizeof(tbuf));
        }
        pp_format_simple_expr(&pp, value, vbuf, sizeof(vbuf));
        if (pp_display_op_type_spelling(ctx, target, ty, sizeof(ty))) {
            snprintf(buf, bufsz, "asr.assign %s to %s : %s", vbuf, tbuf, ty);
        } else {
            snprintf(buf, bufsz, "asr.assign %s to %s", vbuf, tbuf);
        }
        return;
    }
    if (kind == ASR_DIALECT_OP_STMT_DOLOOP) {
        MLIR_RegionHandle head_r = ASR_DoLoopGetHeadRegion(op);
        MLIR_OpHandle head = asr_region_block_op(head_r, 0);
        MLIR_OpHandle v = ASR_do_loop_headVOp(head);
        MLIR_OpHandle start = ASR_do_loop_headStartOp(head);
        MLIR_OpHandle end = ASR_do_loop_headEndOp(head);
        MLIR_OpHandle step = ASR_do_loop_headIncrementOp(head);
        char iv[64];
        char sbuf[160];
        char ebuf[160];
        char stepbuf[160];
        if (ASR_DialectGetOpKind(v) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(ASR_VarV(v), iv, sizeof(iv));
        } else {
            pp_format_simple_expr(&pp, v, iv, sizeof(iv));
        }
        pp_format_simple_expr(&pp, start, sbuf, sizeof(sbuf));
        pp_format_simple_expr(&pp, end, ebuf, sizeof(ebuf));
        if (!pp_is_default_step(step)) {
            pp_format_simple_expr(&pp, step, stepbuf, sizeof(stepbuf));
            snprintf(buf, bufsz, "asr.do_loop %s = %s to %s step %s { ... }",
                iv, sbuf, ebuf, stepbuf);
        } else {
            snprintf(buf, bufsz, "asr.do_loop %s = %s to %s { ... }",
                iv, sbuf, ebuf);
        }
        return;
    }
    if (kind == ASR_DIALECT_OP_STMT_PRINT) {
        MLIR_OpHandle text = ASR_PrintTextOp(op);
        pp_format_print_arg(&pp, text, line, sizeof(line));
        snprintf(buf, bufsz, "asr.print %s", line);
        return;
    }
    if (kind == ASR_DIALECT_OP_STMT_RETURN) {
        snprintf(buf, bufsz, "asr.return");
        return;
    }
    if (kind == ASR_DIALECT_OP_STMT_IF) {
        MLIR_OpHandle test = ASR_IfTestOp(op);
        pp_format_simple_expr(&pp, test, line, sizeof(line));
        snprintf(buf, bufsz, "asr.if %s { ... }", line);
        return;
    }
    if (kind == ASR_DIALECT_OP_STMT_WHILELOOP) {
        MLIR_OpHandle test = ASR_WhileLoopTestOp(op);
        pp_format_simple_expr(&pp, test, line, sizeof(line));
        snprintf(buf, bufsz, "asr.while_loop %s { ... }", line);
        return;
    }
    if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, tbuf, sizeof(tbuf));
        if (pp_display_op_type_spelling(ctx, op, ty, sizeof(ty))) {
            snprintf(buf, bufsz, "asr.variable %s : %s", tbuf, ty);
        } else {
            snprintf(buf, bufsz, "asr.variable %s", tbuf);
        }
        return;
    }

    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (schema && schema->mlir_name) {
        if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM ||
                kind == ASR_DIALECT_OP_SYMBOL_FUNCTION) {
            string name = asr_get_field_str(op, "name");
            pp_format_sym(name, tbuf, sizeof(tbuf));
            snprintf(buf, bufsz, "%s %s", schema->mlir_name, tbuf);
        } else {
            snprintf(buf, bufsz, "%s", schema->mlir_name);
        }
        return;
    }
    snprintf(buf, bufsz, "<unknown op>");
}

string ASR_DialectPrintPretty(MLIR_Context *ctx, MLIR_OpHandle module) {
    ASR_PpCtx pp = {};
    pp.ctx = ctx;
    pp.out = strbuf_make_cap(ctx->arena, 65536);

    strbuf_append_cstr(ctx->arena, &pp.out, "module {\n");
    pp.indent = ASR_PP_INDENT;

    if (MLIR_GetOpType(module) != OP_TYPE_MODULE) {
        return strbuf_to_string(pp.out);
    }

    MLIR_RegionHandle region = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle block = MLIR_GetRegionBlock(region, 0);
    size_t n_ops = MLIR_GetBlockNumOps(block);

    for (size_t i = 0; i < n_ops; ++i) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        if (ASR_DialectGetOpKind(op) == ASR_DIALECT_OP_INVALID) {
            continue;
        }
        pp_stmt(&pp, op);
    }

    strbuf_append_cstr(ctx->arena, &pp.out, "}\n");
    return strbuf_to_string(pp.out);
}
