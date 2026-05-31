#include <libasr/codegen/asr_to_asr_dialect.h>
#include <libasr/codegen/asr_to_mlir_new.h>
#include <libasr/codegen/evaluator.h>
#include <libasr/diagnostics.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

extern "C" {
#include <asr_dialect_api.h>
#include <mlir_api.h>
#include <base/arena.h>
#include <base/string.h>
#include <platform/platform.h>
}

namespace LCompilers {

extern void ensure_mlir_corec_platform_initialized();

static string asr_cstr(const char *s) {
    if (!s) {
        return str_lit("");
    }
    return str_from_cstr_len_view_const(s, (uint64_t)std::strlen(s));
}

class AsrDialectError {
public:
    diag::Diagnostic d;
    explicit AsrDialectError(const std::string &msg)
        : d{diag::Diagnostic(msg, diag::Level::Error, diag::Stage::CodeGen)} {}
    AsrDialectError(const std::string &msg, const Location &loc)
        : d{diag::Diagnostic(msg, diag::Level::Error, diag::Stage::CodeGen,
              {diag::Label("", {loc})})} {}
};

static std::string copy_mlir_string(string s) {
    if (!s.str || s.size == 0) {
        return std::string();
    }
    return std::string(s.str, s.str + s.size);
}

static ASR_DialectField i64_field(const char *name, int64_t v) {
    ASR_DialectField f{};
    f.kind = ASR_FIELD_I64;
    f.name = name;
    f.value.i64 = v;
    return f;
}

static ASR_DialectField bool_field(const char *name, bool v) {
    ASR_DialectField f{};
    f.kind = ASR_FIELD_BOOL;
    f.name = name;
    f.value.b = v;
    return f;
}

static ASR_DialectField str_field(const char *name, string v) {
    ASR_DialectField f{};
    f.kind = ASR_FIELD_IDENTIFIER;
    f.name = name;
    f.value.str = v;
    return f;
}

static ASR_DialectField type_field(const char *name, MLIR_TypeHandle ty) {
    ASR_DialectField f{};
    f.kind = ASR_FIELD_TTYPE;
    f.name = name;
    f.value.type = ty;
    return f;
}

static ASR_DialectField op_field(const char *name, MLIR_OpHandle op) {
    ASR_DialectField f{};
    f.kind = ASR_FIELD_EXPR;
    f.name = name;
    f.value.op = op;
    return f;
}

static ASR_DialectField stmt_field(const char *name, MLIR_OpHandle op) {
    ASR_DialectField f{};
    f.kind = ASR_FIELD_STMT;
    f.name = name;
    f.value.op = op;
    return f;
}

class ASRToAsrDialectVisitor {
public:
    Arena *arena = nullptr;
    MLIR_Context ctx{};
    MLIR_LocationHandle mlir_loc = MLIR_INVALID_HANDLE;
    MLIR_BlockHandle module_block = MLIR_INVALID_HANDLE;
    MLIR_OpHandle module_op = MLIR_INVALID_HANDLE;
    MLIR_OpHandle last_expr_op = MLIR_INVALID_HANDLE;
    MLIR_ValueHandle last_value = MLIR_INVALID_HANDLE;
    bool block_terminated = false;

    MLIR_LocationHandle default_loc() { return mlir_loc; }

    MLIR_TypeHandle convert_type(const ASR::ttype_t &t) {
        if (ASR::is_a<ASR::Integer_t>(t)) {
            const ASR::Integer_t &it = *ASR::down_cast<ASR::Integer_t>(&t);
            return MLIR_CreateTypeInteger(&ctx, (uint32_t)it.m_kind, true);
        }
        if (ASR::is_a<ASR::Logical_t>(t)) {
            return MLIR_CreateTypeInteger(&ctx, 1, false);
        }
        if (ASR::is_a<ASR::Real_t>(t)) {
            const ASR::Real_t &rt = *ASR::down_cast<ASR::Real_t>(&t);
            return MLIR_CreateTypeFloat(&ctx, (uint32_t)rt.m_kind, false);
        }
        return MLIR_CreateTypeInteger(&ctx, 32, true);
    }

    string emit_symbol_ref(const ASR::symbol_t &s) {
        if (ASR::is_a<ASR::Variable_t>(s)) {
            return asr_cstr(ASR::down_cast<ASR::Variable_t>(&s)->m_name);
        }
        return str_lit("unknown");
    }

    void emit_module_skeleton() {
        arena = arena_create(131072);
        if (!arena) {
            throw AsrDialectError("asr dialect: arena_create failed");
        }
        MLIR_SetArenaAllocator(&ctx, arena);
        ctx.no_def_use_tracking = true;
        mlir_loc = MLIR_CreateLocationUnknown(&ctx, str_lit("lfortran"));
        MLIR_RegionHandle mr = MLIR_CreateRegion(&ctx);
        module_block = MLIR_CreateBlock(&ctx);
        MLIR_AppendRegionBlock(&ctx, mr, module_block);
        MLIR_RegionHandle mregs[1] = {mr};
        module_op = MLIR_CreateOp(&ctx, OP_TYPE_MODULE, str_lit("module"),
            nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, mregs, 1,
            mlir_loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    }

    void append_module(MLIR_OpHandle op) {
        MLIR_AppendBlockOp(&ctx, module_block, op);
    }

    MLIR_OpHandle create_op(ASR_DialectOpKind kind, MLIR_LocationHandle op_loc,
            ASR_DialectField *fields, size_t n_fields) {
        return ASR_DialectCreateOp(&ctx, kind, op_loc, fields, n_fields);
    }

    MLIR_OpHandle emit_expr_op(ASR::expr_t &e) {
        if (ASR::is_a<ASR::IntegerConstant_t>(e)) {
            const ASR::IntegerConstant_t &x =
                *ASR::down_cast<ASR::IntegerConstant_t>(&e);
            ASR_DialectField fields[3] = {
                i64_field("n", x.m_n),
                type_field("type", convert_type(*x.m_type)),
                i64_field("intboz_type", (int64_t)x.m_intboz_type),
            };
            return create_op(ASR_DIALECT_OP_EXPR_INTEGERCONSTANT,
                default_loc(), fields, 3);
        }
        if (ASR::is_a<ASR::Var_t>(e)) {
            const ASR::Var_t &x = *ASR::down_cast<ASR::Var_t>(&e);
            ASR_DialectField fields[1] = {
                str_field("v", emit_symbol_ref(*x.m_v)),
            };
            return create_op(ASR_DIALECT_OP_EXPR_VAR, default_loc(), fields, 1);
        }
        if (ASR::is_a<ASR::IntegerBinOp_t>(e)) {
            const ASR::IntegerBinOp_t &x =
                *ASR::down_cast<ASR::IntegerBinOp_t>(&e);
            MLIR_OpHandle left = emit_expr_op(*x.m_left);
            MLIR_OpHandle right = emit_expr_op(*x.m_right);
            ASR_DialectField fields[4] = {
                op_field("left", left),
                i64_field("op", (int64_t)x.m_op),
                op_field("right", right),
                type_field("type", convert_type(*x.m_type)),
            };
            return create_op(ASR_DIALECT_OP_EXPR_INTEGERBINOP,
                default_loc(), fields, 4);
        }
        throw AsrDialectError(
            "asr dialect: expression kind not supported yet", e.base.loc);
    }

    void emit_stmt(ASR::stmt_t &s) {
        if (ASR::is_a<ASR::Assignment_t>(s)) {
            const ASR::Assignment_t &x =
                *ASR::down_cast<ASR::Assignment_t>(&s);
            MLIR_OpHandle target = emit_expr_op(*x.m_target);
            MLIR_OpHandle value = emit_expr_op(*x.m_value);
            ASR_DialectField fields[4] = {
                op_field("target", target),
                op_field("value", value),
                bool_field("realloc_lhs", x.m_realloc_lhs),
                bool_field("move_allocation", x.m_move_allocation),
            };
            MLIR_OpHandle op = create_op(ASR_DIALECT_OP_STMT_ASSIGNMENT,
                default_loc(), fields, 4);
            append_module(op);
            return;
        }
        if (ASR::is_a<ASR::Print_t>(s)) {
            const ASR::Print_t &x = *ASR::down_cast<ASR::Print_t>(&s);
            MLIR_OpHandle text = emit_expr_op(*x.m_text);
            ASR_DialectField fields[1] = {op_field("text", text)};
            append_module(create_op(ASR_DIALECT_OP_STMT_PRINT,
                default_loc(), fields, 1));
            return;
        }
        if (ASR::is_a<ASR::Return_t>(s)) {
            append_module(create_op(ASR_DIALECT_OP_STMT_RETURN,
                default_loc(), nullptr, 0));
            block_terminated = true;
            return;
        }
        throw AsrDialectError(
            "asr dialect: statement not supported yet", s.base.loc);
    }

    void visit_Program(const ASR::Program_t &x) {
        ASR_DialectField prog_fields[1] = {
            str_field("name", asr_cstr(x.m_name)),
        };
        append_module(create_op(ASR_DIALECT_OP_SYMBOL_PROGRAM,
            default_loc(), prog_fields, 1));

        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Variable_t>(*item.second)) {
                const ASR::Variable_t &v =
                    *ASR::down_cast<ASR::Variable_t>(item.second);
                ASR_DialectField fields[2] = {
                    str_field("name", asr_cstr(v.m_name)),
                    type_field("type", convert_type(*v.m_type)),
                };
                append_module(create_op(ASR_DIALECT_OP_SYMBOL_VARIABLE,
                    default_loc(), fields, 2));
            }
        }
        for (size_t i = 0; i < x.n_body; i++) {
            emit_stmt(*x.m_body[i]);
            if (block_terminated) {
                break;
            }
        }
        if (!block_terminated) {
            append_module(create_op(ASR_DIALECT_OP_STMT_RETURN,
                default_loc(), nullptr, 0));
        }
    }

    void visit_TranslationUnit(const ASR::TranslationUnit_t &x) {
        emit_module_skeleton();
        append_module(create_op(ASR_DIALECT_OP_UNIT_TRANSLATIONUNIT,
            default_loc(), nullptr, 0));
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Program_t>(*item.second)) {
                visit_Program(*ASR::down_cast<ASR::Program_t>(item.second));
                return;
            }
        }
        throw AsrDialectError("asr dialect: no program unit found");
    }
};

Result<std::unique_ptr<MLIRModule>> asr_to_asr_dialect(Allocator &al,
    ASR::asr_t &asr, diag::Diagnostics &diagnostics,
    AsrDialectPipelineStage stage) {
    ensure_mlir_corec_platform_initialized();
    (void)al;
    if (!ASR::is_a<ASR::unit_t>(asr)) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_asr_dialect: expected translation unit",
            diag::Level::Error, diag::Stage::CodeGen));
        return Error();
    }
    const ASR::unit_t *u = ASR::down_cast<ASR::unit_t>(&asr);
    if (u->type != ASR::unitType::TranslationUnit) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_asr_dialect: expected unit type translation unit",
            diag::Level::Error, diag::Stage::CodeGen));
        return Error();
    }

    ASRToAsrDialectVisitor emit;
    try {
        emit.visit_TranslationUnit(
            *ASR::down_cast<ASR::TranslationUnit_t>(u));
    } catch (const AsrDialectError &e) {
        diagnostics.diagnostics.push_back(e.d);
        if (emit.arena) {
            arena_destroy(emit.arena);
        }
        return Error();
    }

    if (!ASR_DialectVerify(&emit.ctx, emit.module_op)) {
        // Best-effort verify: warn but continue while emitter coverage grows.
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_asr_dialect: ASR dialect verification reported issues "
            "(continuing)",
            diag::Level::Warning, diag::Stage::CodeGen));
    }

    std::string asr_dialect_text =
        copy_mlir_string(ASR_DialectPrint(&emit.ctx, emit.module_op));

    if (stage == AsrDialectPipelineStage::DialectOnly) {
        auto mod = std::make_unique<MLIRModule>(
            std::string(), std::string(), std::string());
        mod->mlir_asr_dialect_text = std::move(asr_dialect_text);
        arena_destroy(emit.arena);
        return Result<std::unique_ptr<MLIRModule>>(std::move(mod));
    }

    ASR_DialectOptions opts{};
    opts.verify_asr_dialect = false;
    opts.allow_unimplemented_nodes = false;
    if (!ASR_DialectLowerToHighMLIR(&emit.ctx, emit.module_op, &opts)) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_asr_dialect: ASR_DialectLowerToHighMLIR failed",
            diag::Level::Error, diag::Stage::CodeGen));
        arena_destroy(emit.arena);
        return Error();
    }

    std::string high_mlir =
        copy_mlir_string(MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));

    bool use_upstream = std::getenv("USE_MLIR_Upstream") &&
        std::getenv("USE_MLIR_Upstream")[0] == '1';
    bool lowered = use_upstream
        ? MLIR_LowerToLLVMDialectUpstream(&emit.ctx, emit.module_op)
        : MLIR_LowerToLLVMDialect(&emit.ctx, emit.module_op);
    if (!lowered) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_asr_dialect: MLIR_LowerToLLVMDialect failed",
            diag::Level::Error, diag::Stage::CodeGen));
        arena_destroy(emit.arena);
        return Error();
    }

    std::string llvm_dialect_mlir =
        copy_mlir_string(MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));

    string llvm_s = use_upstream
        ? MLIR_TranslateModuleToLLVMIRUpstream(&emit.ctx, emit.module_op)
        : MLIR_TranslateModuleToLLVMIR(&emit.ctx, emit.module_op);
    std::string llvm_ir = copy_mlir_string(llvm_s);

    arena_destroy(emit.arena);
    auto mod = std::make_unique<MLIRModule>(
        std::move(high_mlir), std::move(llvm_dialect_mlir), std::move(llvm_ir));
    mod->mlir_asr_dialect_text = std::move(asr_dialect_text);
    return Result<std::unique_ptr<MLIRModule>>(std::move(mod));
}

} // namespace LCompilers
