#include <libasr/codegen/asr_to_asr_dialect.h>
#include <libasr/codegen/asr_to_mlir_new.h>
#include <libasr/codegen/evaluator.h>
#include <libasr/diagnostics.h>
#include <libasr/asr_utils.h>

#include <cstdint>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include <asr_dialect_api.h>
#include <asr_dialect_emit_registry.h>
#include <mlir_api.h>
#include <base/arena.h>
#include <base/string.h>
#include <platform/platform.h>
}

#include <generated/asr_dialect_api_generated.h>

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

class ASRToAsrDialectVisitor : public ASR::BaseVisitor<ASRToAsrDialectVisitor> {
public:
    typedef MLIR_OpHandle MLIR_ValueHandle;

    Arena *arena = nullptr;
    MLIR_Context ctx{};
    MLIR_LocationHandle mlir_loc = MLIR_INVALID_HANDLE;
    MLIR_BlockHandle module_block = MLIR_INVALID_HANDLE;
    MLIR_OpHandle module_op = MLIR_INVALID_HANDLE;
    MLIR_ValueHandle last_value = MLIR_INVALID_HANDLE;
    bool block_terminated = false;
    // When true, stmt visitors must not append to the module block (e.g. do-loop body).
    bool suppress_module_append = false;

    MLIR_LocationHandle default_loc() { return mlir_loc; }

    MLIR_LocationHandle loc(const Location &l) {
        (void)l;
        return mlir_loc;
    }

    MLIR_TypeHandle convert_type(const ASR::ttype_t &t) {
        if (ASR::is_a<ASR::Array_t>(t)) {
            int64_t len = ASRUtils::get_fixed_size_of_array(
                const_cast<ASR::ttype_t *>(&t));
            if (len <= 0) {
                len = 1;
            }
            int64_t shape[1] = {len};
            ASR::ttype_t *elem = ASRUtils::type_get_past_array(
                const_cast<ASR::ttype_t *>(&t));
            return MLIR_CreateTypeMemref(&ctx, shape, 1, convert_type(*elem));
        }
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

    MLIR_OpHandle *emit_expr_op_array(ASR::expr_t **exprs, size_t n) {
        if (n == 0) {
            return nullptr;
        }
        MLIR_OpHandle *buf = (MLIR_OpHandle *)arena_alloc(
            arena, n * sizeof(MLIR_OpHandle));
        for (size_t i = 0; i < n; ++i) {
            buf[i] = emit_expr(*exprs[i]);
        }
        return buf;
    }

    MLIR_OpHandle *emit_stmt_op_array(ASR::stmt_t **stmts, size_t n) {
        if (n == 0) {
            return nullptr;
        }
        MLIR_OpHandle *buf = (MLIR_OpHandle *)arena_alloc(
            arena, n * sizeof(MLIR_OpHandle));
        for (size_t i = 0; i < n; ++i) {
            buf[i] = emit_stmt(*stmts[i]);
        }
        return buf;
    }

    MLIR_OpHandle *emit_array_index_op_array(const ASR::array_index_t *indices,
            size_t n) {
        if (n == 0 || !indices) {
            return nullptr;
        }
        MLIR_OpHandle *buf = (MLIR_OpHandle *)arena_alloc(
            arena, n * sizeof(MLIR_OpHandle));
        for (size_t i = 0; i < n; ++i) {
            buf[i] = emit_array_index(indices[i]);
        }
        return buf;
    }

    MLIR_OpHandle *emit_product_op_array(const void *nodes, size_t n) {
        (void)nodes;
        (void)n;
        return nullptr;
    }

    string emit_symbol_ref(const ASR::symbol_t &s) {
        if (ASR::is_a<ASR::Variable_t>(s)) {
            return asr_cstr(ASR::down_cast<ASR::Variable_t>(&s)->m_name);
        }
        if (ASR::is_a<ASR::Function_t>(s)) {
            return asr_cstr(ASR::down_cast<ASR::Function_t>(&s)->m_name);
        }
        if (ASR::is_a<ASR::Program_t>(s)) {
            return asr_cstr(ASR::down_cast<ASR::Program_t>(&s)->m_name);
        }
        return str_lit("unknown");
    }

    string emit_identifier_seq(char **ids, size_t n) {
        (void)ids;
        (void)n;
        return str_lit("");
    }

    string emit_symbol_seq_ref(ASR::symbol_t **syms, size_t n) {
        (void)syms;
        (void)n;
        return str_lit("");
    }

    MLIR_ValueHandle emit_product_value(ASR::asr_t &node) {
        (void)node;
        return MLIR_INVALID_HANDLE;
    }

    MLIR_ValueHandle emit_product_seq_value(ASR::asr_t **nodes, size_t n) {
        (void)nodes;
        (void)n;
        return MLIR_INVALID_HANDLE;
    }

    MLIR_TypeHandle emit_type_seq_value(ASR::ttype_t **types, size_t n) {
        (void)types;
        (void)n;
        return MLIR_INVALID_HANDLE;
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
        if (op != MLIR_INVALID_HANDLE) {
            MLIR_AppendBlockOp(&ctx, module_block, op);
        }
    }

    void append_current_stmt(MLIR_OpHandle op) {
        if (!suppress_module_append) {
            append_module(op);
        }
        if (last_value != MLIR_INVALID_HANDLE &&
            ASR_DialectGetOpKind(last_value) == ASR_DIALECT_OP_STMT_RETURN) {
            block_terminated = true;
        }
        if (last_value != MLIR_INVALID_HANDLE &&
            ASR_DialectGetOpKind(last_value) == ASR_DIALECT_OP_STMT_ERRORSTOP) {
            block_terminated = true;
        }
    }

    MLIR_ValueHandle emit_expr(ASR::expr_t &e) {
        last_value = MLIR_INVALID_HANDLE;
        ASR::BaseVisitor<ASRToAsrDialectVisitor>::visit_expr(e);
        return last_value;
    }

    MLIR_ValueHandle emit_stmt(ASR::stmt_t &s) {
        last_value = MLIR_INVALID_HANDLE;
        ASR::BaseVisitor<ASRToAsrDialectVisitor>::visit_stmt(s);
        return last_value;
    }

    MLIR_ValueHandle emit_expr_seq_value(ASR::expr_t **exprs, size_t n) {
        MLIR_ValueHandle last = MLIR_INVALID_HANDLE;
        for (size_t i = 0; i < n; ++i) {
            last = emit_expr(*exprs[i]);
        }
        return last;
    }

    MLIR_ValueHandle emit_stmt_seq_value(ASR::stmt_t **stmts, size_t n) {
        MLIR_ValueHandle last = MLIR_INVALID_HANDLE;
        for (size_t i = 0; i < n; ++i) {
            last = emit_stmt(*stmts[i]);
        }
        return last;
    }

    MLIR_ValueHandle emit_do_loop_head(const ASR::do_loop_head_t &h) {
        MLIR_ValueHandle v = MLIR_INVALID_HANDLE;
        MLIR_ValueHandle start = MLIR_INVALID_HANDLE;
        MLIR_ValueHandle end = MLIR_INVALID_HANDLE;
        MLIR_ValueHandle increment = MLIR_INVALID_HANDLE;
        if (h.m_v) {
            v = emit_expr(*h.m_v);
        }
        if (h.m_start) {
            start = emit_expr(*h.m_start);
        }
        if (h.m_end) {
            end = emit_expr(*h.m_end);
        }
        if (h.m_increment) {
            increment = emit_expr(*h.m_increment);
        }
        return ASR_Createdo_loop_headOp(&ctx, default_loc(), v, start, end, increment);
    }

    MLIR_ValueHandle emit_array_index(const ASR::array_index_t &idx) {
        MLIR_ValueHandle left = MLIR_INVALID_HANDLE;
        MLIR_ValueHandle right = MLIR_INVALID_HANDLE;
        MLIR_ValueHandle step = MLIR_INVALID_HANDLE;
        if (idx.m_left) {
            left = emit_expr(*idx.m_left);
        }
        if (idx.m_right) {
            right = emit_expr(*idx.m_right);
        }
        if (idx.m_step) {
            step = emit_expr(*idx.m_step);
        }
        return ASR_Createarray_indexOp(&ctx, default_loc(), left, right, step);
    }

    MLIR_ValueHandle emit_print_op(ASR::expr_t &text_expr) {
        MLIR_ValueHandle text = emit_expr(text_expr);
        last_value = ASR_CreatePrintOp(&ctx, default_loc(), text);
        return last_value;
    }

    void emit_print_exprs(ASR::expr_t &text_expr) {
        if (ASR::is_a<ASR::StringFormat_t>(text_expr)) {
            const ASR::StringFormat_t &sf =
                *ASR::down_cast<ASR::StringFormat_t>(&text_expr);
            for (size_t i = 0; i < sf.n_args; i++) {
                append_current_stmt(emit_print_op(*sf.m_args[i]));
            }
            return;
        }
        append_current_stmt(emit_print_op(text_expr));
    }

    void visit_Print(const ASR::Print_t &x) {
        if (!x.m_text) {
            throw AsrDialectError(
                "asr dialect: print with no format expression", x.base.base.loc);
        }
        emit_print_exprs(*x.m_text);
    }

    void visit_StringFormat(const ASR::StringFormat_t &x) {
        MLIR_ValueHandle fmt = MLIR_INVALID_HANDLE;
        if (x.m_fmt) {
            fmt = emit_expr(*x.m_fmt);
        }
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, x.n_args);
        MLIR_TypeHandle type = convert_type(*x.m_type);
        MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) {
            value = emit_expr(*x.m_value);
        }
        MLIR_LocationHandle op_loc = loc(x.base.base.loc);
        last_value = ASR_CreateStringFormatOp(&ctx, op_loc, fmt, args, x.n_args,
            (int64_t)x.m_kind, type, value);
        MLIR_TypeHandle i64_ty = MLIR_CreateTypeInteger(&ctx, 64, false);
        MLIR_AttributeHandle n_attr = MLIR_CreateAttributeInteger(
            &ctx, str_lit("asr.f.n_args"), (int64_t)x.n_args, i64_ty);
        MLIR_AppendOpAttribute(&ctx, last_value, n_attr);
    }

    void visit_FileWrite(const ASR::FileWrite_t &x) {
        bool is_stdout = false;
        if (x.m_unit) {
            int unit_value = -1;
            if (ASRUtils::extract_value(x.m_unit, unit_value) && unit_value == 6) {
                is_stdout = true;
            }
        } else {
            is_stdout = true;
        }
        if (!is_stdout) {
            throw AsrDialectError(
                "asr dialect: FileWrite only supports default output unit (6)",
                x.base.base.loc);
        }
        if (x.n_values == 0) {
            return;
        }
        if (x.n_values == 1) {
            emit_print_exprs(*x.m_values[0]);
            return;
        }
        for (size_t i = 0; i < x.n_values; i++) {
            append_current_stmt(emit_print_op(*x.m_values[i]));
        }
    }

    void visit_DoLoop(const ASR::DoLoop_t &x) {
        if (x.n_orelse > 0) {
            throw AsrDialectError(
                "asr dialect: do-loop else not supported yet", x.base.base.loc);
        }
        std::vector<MLIR_OpHandle> body_ops;
        body_ops.reserve(x.n_body);
        suppress_module_append = true;
        for (size_t i = 0; i < x.n_body; i++) {
            MLIR_OpHandle st = emit_stmt(*x.m_body[i]);
            if (st != MLIR_INVALID_HANDLE) {
                body_ops.push_back(st);
            }
        }
        suppress_module_append = false;
        MLIR_ValueHandle head = emit_do_loop_head(x.m_head);
        MLIR_OpHandle *body_ptr = nullptr;
        size_t n_body = body_ops.size();
        if (n_body > 0) {
            body_ptr = (MLIR_OpHandle *)arena_alloc(
                arena, n_body * sizeof(MLIR_OpHandle));
            for (size_t i = 0; i < n_body; ++i) {
                body_ptr[i] = body_ops[i];
            }
        }
        last_value = ASR_CreateDoLoopOp(&ctx, default_loc(),
            asr_cstr(x.m_name), head, body_ptr, n_body, nullptr, 0);
        append_current_stmt(last_value);
        if (body_ptr) {
            ASR_DialectEmitRegistryAddDoLoopBody(
                last_value, body_ptr, n_body);
        }
    }

    void visit_Variable(const ASR::Variable_t &v) {
        MLIR_ValueHandle symbolic_value = MLIR_INVALID_HANDLE;
        MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (v.m_symbolic_value) {
            symbolic_value = emit_expr(*v.m_symbolic_value);
        }
        if (v.m_value) {
            value = emit_expr(*v.m_value);
        }
        string type_decl = str_lit("");
        if (v.m_type_declaration) {
            type_decl = emit_symbol_ref(*v.m_type_declaration);
        }
        MLIR_TypeHandle var_ty = convert_type(*v.m_type);
        last_value = ASR_CreateVariableOp(&ctx, default_loc(),
            MLIR_INVALID_HANDLE, asr_cstr(v.m_name), str_lit(""),
            (int64_t)v.m_intent, symbolic_value, value, (int64_t)v.m_storage,
            var_ty, type_decl, (int64_t)v.m_abi,
            (int64_t)v.m_access, (int64_t)v.m_presence, v.m_value_attr,
            v.m_target_attr, v.m_contiguous_attr,
            v.m_bindc_name ? asr_cstr(v.m_bindc_name) : str_lit(""),
            v.m_is_volatile, v.m_is_protected, (int64_t)v.m_pass_attr,
            v.m_self_argument ? asr_cstr(v.m_self_argument) : str_lit(""),
            nullptr, 0);
        if (ASR::is_a<ASR::Array_t>(*v.m_type)) {
            int64_t arr_len = ASRUtils::get_fixed_size_of_array(v.m_type);
            MLIR_TypeHandle i64_ty = MLIR_CreateTypeInteger(&ctx, 64, false);
            MLIR_AttributeHandle len_attr = MLIR_CreateAttributeInteger(
                &ctx, str_lit("asr.f.array_len"), arr_len, i64_ty);
            MLIR_AppendOpAttribute(&ctx, last_value, len_attr);
        }
        append_module(last_value);
    }

    void append_seq_n_args_attr(MLIR_OpHandle op, size_t n) {
        MLIR_TypeHandle i64_ty = MLIR_CreateTypeInteger(&ctx, 64, false);
        MLIR_AttributeHandle n_attr = MLIR_CreateAttributeInteger(
            &ctx, str_lit("asr.f.n_args"), (int64_t)n, i64_ty);
        MLIR_AppendOpAttribute(&ctx, op, n_attr);
    }

    void visit_ArrayConstructor(const ASR::ArrayConstructor_t &x) {
        size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, n_args);
        MLIR_TypeHandle type = convert_type(*x.m_type);
        MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) {
            value = emit_expr(*x.m_value);
        }
        int64_t storage_format = (int64_t)x.m_storage_format;
        MLIR_ValueHandle struct_var = MLIR_INVALID_HANDLE;
        if (x.m_struct_var) {
            struct_var = emit_expr(*x.m_struct_var);
        }
        last_value = ASR_CreateArrayConstructorOp(&ctx, default_loc(), args, n_args,
            type, value, storage_format, struct_var);
        append_seq_n_args_attr(last_value, n_args);
    }

    void visit_ArrayItem(const ASR::ArrayItem_t &x) {
        MLIR_ValueHandle v = emit_expr(*x.m_v);
        size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_array_index_op_array(x.m_args, n_args);
        MLIR_TypeHandle type = convert_type(*x.m_type);
        int64_t storage_format = (int64_t)x.m_storage_format;
        MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) {
            value = emit_expr(*x.m_value);
        }
        last_value = ASR_CreateArrayItemOp(&ctx, default_loc(), v, args, n_args,
            type, storage_format, value);
        append_seq_n_args_attr(last_value, n_args);
    }

#include <libasr/codegen/generated/asr_to_asr_dialect_visitor.inc>

    MLIR_OpHandle program_op = MLIR_INVALID_HANDLE;

    void visit_Program(const ASR::Program_t &x) {
        program_op = ASR_CreateProgramOp(&ctx, default_loc(),
            MLIR_INVALID_HANDLE, asr_cstr(x.m_name), str_lit(""),
            nullptr, 0, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE);
        last_value = program_op;
        append_module(program_op);

        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Variable_t>(*item.second)) {
                visit_Variable(*ASR::down_cast<ASR::Variable_t>(item.second));
            }
        }
        for (size_t i = 0; i < x.n_body; i++) {
            emit_stmt(*x.m_body[i]);
            if (block_terminated) {
                break;
            }
        }
        if (!block_terminated) {
            last_value = ASR_CreateReturnOp(&ctx, default_loc());
            append_current_stmt(last_value);
        }
        last_value = MLIR_INVALID_HANDLE;
    }

    void visit_TranslationUnit(const ASR::TranslationUnit_t &x) {
        emit_module_skeleton();
        ASR_DialectEmitRegistryClear();
        program_op = MLIR_INVALID_HANDLE;
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Program_t>(*item.second)) {
                visit_Program(*ASR::down_cast<ASR::Program_t>(item.second));
                break;
            }
        }
        if (program_op == MLIR_INVALID_HANDLE) {
            throw AsrDialectError("asr dialect: no program unit found");
        }
        MLIR_OpHandle items[1] = {program_op};
        last_value = ASR_CreateTranslationUnitOp(&ctx, default_loc(),
            MLIR_INVALID_HANDLE, items, 1);
        MLIR_InsertBlockOpAtIndex(&ctx, module_block, last_value, 0);
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
        ASR_DialectEmitRegistryClear();
        if (emit.arena) {
            arena_destroy(emit.arena);
        }
        return Error();
    }

    if (!ASR_DialectVerify(&emit.ctx, emit.module_op)) {
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
        ASR_DialectEmitRegistryClear();
        arena_destroy(emit.arena);
        return Result<std::unique_ptr<MLIRModule>>(std::move(mod));
    }

    ASR_DialectOptions opts{};
    opts.verify_asr_dialect = false;
    opts.allow_unimplemented_nodes = true;
    if (!ASR_DialectLowerToHighMLIR(&emit.ctx, emit.module_op, &opts)) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_asr_dialect: ASR_DialectLowerToHighMLIR failed",
            diag::Level::Error, diag::Stage::CodeGen));
        ASR_DialectEmitRegistryClear();
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
        ASR_DialectEmitRegistryClear();
        arena_destroy(emit.arena);
        return Error();
    }

    std::string llvm_dialect_mlir =
        copy_mlir_string(MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));

    string llvm_s = use_upstream
        ? MLIR_TranslateModuleToLLVMIRUpstream(&emit.ctx, emit.module_op)
        : MLIR_TranslateModuleToLLVMIR(&emit.ctx, emit.module_op);
    std::string llvm_ir = copy_mlir_string(llvm_s);

    ASR_DialectEmitRegistryClear();
    arena_destroy(emit.arena);
    auto mod = std::make_unique<MLIRModule>(
        std::move(high_mlir), std::move(llvm_dialect_mlir), std::move(llvm_ir));
    mod->mlir_asr_dialect_text = std::move(asr_dialect_text);
    return Result<std::unique_ptr<MLIRModule>>(std::move(mod));
}

} // namespace LCompilers
