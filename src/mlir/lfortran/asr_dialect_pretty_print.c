// Statement-first ASR dialect pretty printer (3_lfortran/alternative_2.md).
#include "asr_dialect_api.h"
#include "asr_dialect_emit_registry.h"
#include "asr_dialect_fields.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <base/strbuf.h>
#include <base/string.h>

#include "generated/asr_dialect_enum_print.inc"

#define ASR_PP_INDENT 2
#define ASR_PP_UNRESOLVED "<unresolved expr>"

typedef struct {
    MLIR_Context *ctx;
    strbuf out;
    int indent;
    bool in_program;
    bool in_translation_unit;
} ASR_PpCtx;

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
    if (!name.str || name.size == 0) {
        snprintf(buf, bufsz, "@?");
        return;
    }
    snprintf(buf, bufsz, "@%.*s", (int)name.size, name.str);
}

/* Returns true when a type suffix should be printed. Never emits !asr.unknown. */
static bool pp_type_spelling(MLIR_TypeHandle ty, char *buf, size_t bufsz) {
    if (ty == MLIR_INVALID_HANDLE) {
        return false;
    }
    if (MLIR_IsTypeInteger(ty)) {
        snprintf(buf, bufsz, "i32");
        return true;
    }
    if (MLIR_IsTypeFloat(ty)) {
        snprintf(buf, bufsz, "real(8)");
        return true;
    }
    return false;
}

static MLIR_TypeHandle pp_expr_result_type(MLIR_OpHandle op) {
    MLIR_TypeHandle ty = asr_get_field_type(op, "type");
    if (ty != MLIR_INVALID_HANDLE) {
        return ty;
    }
    return MLIR_INVALID_HANDLE;
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

    ASR_DialectOpKind kind = ASR_DialectGetOpKind(op);
    switch (kind) {
        case ASR_DIALECT_OP_EXPR_INTEGERCONSTANT:
            snprintf(buf, bufsz, "%" PRId64, asr_get_field_i64(op, "n", 0));
            return;
        case ASR_DIALECT_OP_EXPR_VAR: {
            string sym = asr_get_field_str(op, "v");
            pp_format_sym(sym, buf, bufsz);
            return;
        }
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
        default:
            break;
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

static bool pp_variable_field_is_default(const char *fname, MLIR_OpHandle op) {
    if (strcmp(fname, "name") == 0) {
        return true;
    }
    if (strcmp(fname, "intent") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "storage") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "abi") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "access") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "presence") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "pass_attr") == 0) {
        return asr_get_field_i64(op, fname, 0) == 0;
    }
    if (strcmp(fname, "value_attr") == 0 || strcmp(fname, "target_attr") == 0 ||
            strcmp(fname, "contiguous_attr") == 0 || strcmp(fname, "is_volatile") == 0 ||
            strcmp(fname, "is_protected") == 0) {
        return !asr_get_field_bool(op, fname, false);
    }
    if (strcmp(fname, "bindc_name") == 0 || strcmp(fname, "self_argument") == 0) {
        return asr_get_field_str(op, fname).size == 0;
    }
    return false;
}

static bool pp_should_elide_field(
        ASR_DialectOpKind kind, const ASR_DialectFieldDesc *fd, MLIR_OpHandle op) {
    if (!fd) {
        return true;
    }
    if (asr_get_field_attr(op, fd->name) == MLIR_INVALID_HANDLE) {
        return true;
    }
    if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
        return pp_variable_field_is_default(fd->name, op);
    }
    if (fd->kind == ASR_FIELD_TTYPE) {
        return true;
    }
    if (fd->kind == ASR_FIELD_BOOL || fd->kind == ASR_FIELD_BOOL_OPT) {
        return !asr_get_field_bool(op, fd->name, false);
    }
    if (fd->kind == ASR_FIELD_I64 || fd->kind == ASR_FIELD_I64_OPT) {
        return asr_get_field_i64(op, fd->name, 0) == 0;
    }
    if (fd->kind == ASR_FIELD_IDENTIFIER || fd->kind == ASR_FIELD_IDENTIFIER_OPT ||
            fd->kind == ASR_FIELD_STRING || fd->kind == ASR_FIELD_STRING_OPT) {
        return asr_get_field_str(op, fd->name).size == 0;
    }
    return false;
}

static const char *pp_enum_field_name(
        ASR_DialectOpKind kind, const char *field, int64_t v) {
    if (strcmp(field, "intent") == 0) {
        return asr_enum_intent_name(v);
    }
    if (strcmp(field, "storage") == 0) {
        return asr_enum_storage_type_name(v);
    }
    if (strcmp(field, "access") == 0) {
        return asr_enum_access_name(v);
    }
    if (strcmp(field, "abi") == 0) {
        return asr_enum_abi_name(v);
    }
    if (strcmp(field, "presence") == 0) {
        return asr_enum_presence_name(v);
    }
    if (strcmp(field, "pass_attr") == 0) {
        return asr_enum_pass_attr_name(v);
    }
    (void)kind;
    return NULL;
}

static void pp_print_variable_attrs(ASR_PpCtx *pp, MLIR_OpHandle op) {
    const ASR_DialectOpSchema *schema =
        ASR_DialectLookupSchema(ASR_DIALECT_OP_SYMBOL_VARIABLE);
    if (!schema) {
        return;
    }
    for (size_t i = 0; i < schema->n_fields; ++i) {
        const ASR_DialectFieldDesc *fd = &schema->fields[i];
        const char *fname = fd->name;
        if (strcmp(fname, "name") == 0 || strcmp(fname, "type") == 0) {
            continue;
        }
        if (pp_should_elide_field(ASR_DIALECT_OP_SYMBOL_VARIABLE, fd, op)) {
            continue;
        }
        if (fd->kind == ASR_FIELD_BOOL || fd->kind == ASR_FIELD_BOOL_OPT) {
            pp_fmt(pp, "  %s = true", fname);
            continue;
        }
        if (fd->kind == ASR_FIELD_I64 || fd->kind == ASR_FIELD_I64_OPT) {
            const char *en = pp_enum_field_name(ASR_DIALECT_OP_SYMBOL_VARIABLE, fname,
                asr_get_field_i64(op, fname, 0));
            if (en) {
                pp_fmt(pp, "  %s = %s", fname, en);
            }
            continue;
        }
        if (fd->kind == ASR_FIELD_STRING || fd->kind == ASR_FIELD_STRING_OPT ||
                fd->kind == ASR_FIELD_IDENTIFIER || fd->kind == ASR_FIELD_IDENTIFIER_OPT) {
            string s = asr_get_field_str(op, fname);
            pp_fmt(pp, "  %s = \"%.*s\"", fname, (int)s.size, s.str ? s.str : "");
        }
    }
}

static void pp_stmt(ASR_PpCtx *pp, MLIR_OpHandle op);
static void pp_stmt_block(ASR_PpCtx *pp, MLIR_OpHandle op);

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
        pp->in_translation_unit = true;
        return;
    }

    if (kind == ASR_DIALECT_OP_SYMBOL_PROGRAM) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, buf, sizeof(buf));
        pp_fmt(pp, "asr.program %s {", buf);
        pp->indent += ASR_PP_INDENT;
        pp->in_program = true;
        return;
    }

    if (kind == ASR_DIALECT_OP_SYMBOL_VARIABLE) {
        string name = asr_get_field_str(op, "name");
        pp_format_sym(name, buf, sizeof(buf));
        if (pp_type_spelling(asr_get_field_type(op, "type"), ty, sizeof(ty))) {
            pp_fmt(pp, "asr.variable %s : %s", buf, ty);
        } else {
            pp_fmt(pp, "asr.variable %s", buf);
        }
        pp_print_variable_attrs(pp, op);
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_ASSIGNMENT) {
        MLIR_OpHandle target = asr_get_field_op(op, "target");
        MLIR_OpHandle value = asr_get_field_op(op, "value");
        char tbuf[160];
        char vbuf[160];
        if (ASR_DialectGetOpKind(target) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(asr_get_field_str(target, "v"), tbuf, sizeof(tbuf));
        } else {
            pp_expr(pp, target, tbuf, sizeof(tbuf));
        }
        pp_expr(pp, value, vbuf, sizeof(vbuf));
        if (pp_type_spelling(pp_expr_result_type(value), ty, sizeof(ty))) {
            pp_fmt(pp, "asr.assign %s = %s : %s", tbuf, vbuf, ty);
        } else {
            pp_fmt(pp, "asr.assign %s = %s", tbuf, vbuf);
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_DOLOOP) {
        MLIR_OpHandle head = asr_get_field_op(op, "head");
        MLIR_OpHandle body = asr_get_field_op(op, "body");
        MLIR_OpHandle orelse = asr_get_field_op(op, "orelse");
        MLIR_OpHandle v = asr_get_field_op(head, "v");
        MLIR_OpHandle start = asr_get_field_op(head, "start");
        MLIR_OpHandle end = asr_get_field_op(head, "end");
        MLIR_OpHandle step = asr_get_field_op(head, "increment");
        char iv[64];
        char sbuf[160];
        char ebuf[160];
        char stepbuf[160];

        if (ASR_DialectGetOpKind(v) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(asr_get_field_str(v, "v"), iv, sizeof(iv));
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
        {
            size_t n_body = ASR_DialectEmitRegistryDoLoopBodyCount(op);
            if (n_body > 0) {
                for (size_t bi = 0; bi < n_body; ++bi) {
                    MLIR_OpHandle body_stmt =
                        ASR_DialectEmitRegistryDoLoopBodyOp(op, bi);
                    pp_stmt_block(pp, body_stmt);
                }
            } else if (body != MLIR_INVALID_HANDLE) {
                pp_stmt_block(pp, body);
            }
        }
        pp->indent -= ASR_PP_INDENT;

        if (orelse != MLIR_INVALID_HANDLE) {
            pp_line(pp, "} else {");
            pp->indent += ASR_PP_INDENT;
            pp_stmt_block(pp, orelse);
            pp->indent -= ASR_PP_INDENT;
            pp_line(pp, "}");
        } else {
            pp_line(pp, "}");
        }
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_PRINT) {
        MLIR_OpHandle text = asr_get_field_op(op, "text");
        if (ASR_DialectGetOpKind(text) == ASR_DIALECT_OP_EXPR_VAR) {
            pp_format_sym(asr_get_field_str(text, "v"), buf, sizeof(buf));
        } else {
            pp_expr(pp, text, buf, sizeof(buf));
        }
        pp_fmt(pp, "asr.print %s", buf);
        return;
    }

    if (kind == ASR_DIALECT_OP_STMT_RETURN) {
        pp_line(pp, "asr.return");
        return;
    }

    /* Expression ops are not printed at module level — only inline in statements. */
    if (kind == ASR_DIALECT_OP_EXPR_INTEGERCONSTANT ||
            kind == ASR_DIALECT_OP_EXPR_INTEGERBINOP ||
            kind == ASR_DIALECT_OP_EXPR_VAR) {
        return;
    }

    const ASR_DialectOpSchema *schema = ASR_DialectLookupSchema(kind);
    if (schema && schema->mlir_name) {
        pp_fmt(pp, "%s {", schema->mlir_name);
        pp->indent += ASR_PP_INDENT;
        for (size_t i = 0; i < schema->n_fields; ++i) {
            const ASR_DialectFieldDesc *fd = &schema->fields[i];
            if (pp_should_elide_field(kind, fd, op)) {
                continue;
            }
            if (fd->kind == ASR_FIELD_EXPR || fd->kind == ASR_FIELD_STMT ||
                    fd->kind == ASR_FIELD_OP) {
                char ebuf[160];
                pp_expr(pp, asr_get_field_op(op, fd->name), ebuf, sizeof(ebuf));
                pp_fmt(pp, "  %s = %s", fd->name, ebuf);
            }
        }
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
    }
}

static void pp_stmt_block(ASR_PpCtx *pp, MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return;
    }
    pp_stmt(pp, op);
}

static void pp_close_scopes(ASR_PpCtx *pp) {
    if (pp->in_program) {
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        pp->in_program = false;
    }
    if (pp->in_translation_unit) {
        pp->indent -= ASR_PP_INDENT;
        pp_line(pp, "}");
        pp->in_translation_unit = false;
    }
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

    pp_close_scopes(&pp);
    strbuf_append_cstr(ctx->arena, &pp.out, "}\n");
    return strbuf_to_string(pp.out);
}
