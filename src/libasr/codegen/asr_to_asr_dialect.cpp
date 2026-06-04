#include <libasr/codegen/asr_to_asr_dialect.h>

#include <cstring>

#include <generated/asr_dialect_api_generated.h>

namespace LCompilers {

static string asr_cstr(const char *s) {
    if (!s) {
        return str_lit("");
    }
    return str_from_cstr_len_view_const(s, (uint64_t)std::strlen(s));
}

MLIR_LocationHandle ASRToAsrDialectVisitor::default_loc() {
    return mlir_loc;
}

MLIR_LocationHandle ASRToAsrDialectVisitor::loc(const Location &l) {
    (void)l;
    return mlir_loc;
}

uint32_t ASRToAsrDialectVisitor::asr_kind_to_bits(int64_t kind) {
    switch (kind) {
        case 1: return 8;
        case 2: return 16;
        case 4: return 32;
        case 8: return 64;
        default: return 32;
    }
}

MLIR_TypeHandle ASRToAsrDialectVisitor::asr_dialect_type(const ASR::ttype_t &t) {
    if (ASR::is_a<ASR::Array_t>(t)) {
        int64_t len = ASRUtils::get_fixed_size_of_array(
            const_cast<ASR::ttype_t *>(&t));
        if (len <= 0) {
            len = 1;
        }
        int64_t shape[1] = {len};
        ASR::ttype_t *elem = ASRUtils::type_get_past_array(
            const_cast<ASR::ttype_t *>(&t));
        MLIR_TypeHandle elem_ty = asr_dialect_type(*elem);
        MLIR_TypeHandle memref_ty =
            MLIR_CreateTypeMemref(&ctx, shape, 1, elem_ty);
        int64_t elem_kind = 4;
        if (ASR::is_a<ASR::Integer_t>(*elem)) {
            elem_kind = ASR::down_cast<ASR::Integer_t>(elem)->m_kind;
        }
        ASR_ModuleStorageSetTypeInfo(memref_ty, elem_kind, true, len);
        return memref_ty;
    }
    if (ASR::is_a<ASR::Integer_t>(t)) {
        const ASR::Integer_t &it = *ASR::down_cast<ASR::Integer_t>(&t);
        MLIR_TypeHandle ty = MLIR_CreateTypeInteger(
            &ctx, asr_kind_to_bits(it.m_kind), true);
        ASR_ModuleStorageSetTypeInfo(ty, it.m_kind, false, 0);
        return ty;
    }
    if (ASR::is_a<ASR::Logical_t>(t)) {
        MLIR_TypeHandle ty = MLIR_CreateTypeInteger(&ctx, 1, false);
        ASR_ModuleStorageSetTypeInfo(ty, 0, false, 0);
        return ty;
    }
    if (ASR::is_a<ASR::Real_t>(t)) {
        const ASR::Real_t &rt = *ASR::down_cast<ASR::Real_t>(&t);
        MLIR_TypeHandle ty = MLIR_CreateTypeFloat(
            &ctx, asr_kind_to_bits(rt.m_kind), false);
        ASR_ModuleStorageSetTypeInfo(ty, rt.m_kind, false, 0);
        return ty;
    }
    MLIR_TypeHandle ty = MLIR_CreateTypeInteger(&ctx, 32, true);
    ASR_ModuleStorageSetTypeInfo(ty, 4, false, 0);
    return ty;
}

MLIR_TypeHandle ASRToAsrDialectVisitor::convert_type(const ASR::ttype_t &t) {
    return asr_dialect_type(t);
}

MLIR_OpHandle *ASRToAsrDialectVisitor::emit_expr_op_array(ASR::expr_t **exprs, size_t n) {
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

MLIR_OpHandle *ASRToAsrDialectVisitor::emit_stmt_op_array(ASR::stmt_t **stmts, size_t n) {
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

MLIR_OpHandle *ASRToAsrDialectVisitor::emit_array_index_op_array(
        const ASR::array_index_t *indices, size_t n) {
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

MLIR_OpHandle *ASRToAsrDialectVisitor::emit_product_op_array(const void *nodes, size_t n) {
    (void)nodes;
    (void)n;
    return nullptr;
}

string ASRToAsrDialectVisitor::emit_symbol_ref(const ASR::symbol_t &s) {
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

string ASRToAsrDialectVisitor::emit_identifier_seq(char **ids, size_t n) {
    if (n == 0 || !ids) {
        return str_lit("");
    }
    size_t total = 0;
    for (size_t i = 0; i < n; ++i) {
        if (ids[i]) {
            total += std::strlen(ids[i]);
        }
        total += 2;
    }
    char *buf = (char *)arena_alloc(arena, total + 1);
    size_t pos = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            buf[pos++] = ',';
            buf[pos++] = ' ';
        }
        if (ids[i]) {
            size_t len = std::strlen(ids[i]);
            std::memcpy(buf + pos, ids[i], len);
            pos += len;
        }
    }
    buf[pos] = '\0';
    return str_from_cstr_view(buf);
}

string ASRToAsrDialectVisitor::emit_symbol_seq_ref(ASR::symbol_t **syms, size_t n) {
    (void)syms;
    (void)n;
    return str_lit("");
}

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_product_value(ASR::asr_t &node) {
    (void)node;
    return MLIR_INVALID_HANDLE;
}

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_product_seq_value(ASR::asr_t **nodes, size_t n) {
    (void)nodes;
    (void)n;
    return MLIR_INVALID_HANDLE;
}

MLIR_TypeHandle ASRToAsrDialectVisitor::emit_type_seq_value(ASR::ttype_t **types, size_t n) {
    (void)types;
    (void)n;
    return MLIR_INVALID_HANDLE;
}

void ASRToAsrDialectVisitor::emit_module_skeleton() {
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

void ASRToAsrDialectVisitor::append_module(MLIR_OpHandle op) {
    if (op != MLIR_INVALID_HANDLE) {
        MLIR_AppendBlockOp(&ctx, module_block, op);
    }
}

MLIR_OpHandle ASRToAsrDialectVisitor::create_scope_container(
        const char *region_name, const std::vector<MLIR_OpHandle> &ops) {
    string opname = asr_cstr(region_name);
    MLIR_OpHandle container = MLIR_CreateOp(&ctx, OP_TYPE_UNREGISTERED, opname,
        nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
        default_loc(), MLIR_INVALID_HANDLE, str_lit(""), -1);
    if (container == MLIR_INVALID_HANDLE) {
        return MLIR_INVALID_HANDLE;
    }
    if (!ops.empty()) {
        MLIR_OpHandle *buf = (MLIR_OpHandle *)arena_alloc(
            arena, ops.size() * sizeof(MLIR_OpHandle));
        for (size_t i = 0; i < ops.size(); ++i) {
            buf[i] = ops[i];
        }
        ASR_ModuleStorageSetFieldOpSeq(container, "ops", buf, ops.size());
    }
    MLIR_TypeHandle i64_ty = MLIR_CreateTypeInteger(&ctx, 64, false);
    MLIR_AppendOpAttribute(&ctx, container, MLIR_CreateAttributeInteger(
        &ctx, str_lit("asr.n_ops"), (int64_t)ops.size(), i64_ty));
    return container;
}

void ASRToAsrDialectVisitor::attach_variable_type_attrs(
        MLIR_OpHandle var_op, const ASR::ttype_t &t) {
    int64_t asr_kind = 4;
    int64_t array_len = 0;
    bool is_array = ASR::is_a<ASR::Array_t>(t);
    ASR::ttype_t *elem = const_cast<ASR::ttype_t *>(&t);
    if (is_array) {
        array_len = ASRUtils::get_fixed_size_of_array(elem);
        elem = ASRUtils::type_get_past_array(elem);
    }
    if (ASR::is_a<ASR::Integer_t>(*elem)) {
        asr_kind = ASR::down_cast<ASR::Integer_t>(elem)->m_kind;
    } else if (ASR::is_a<ASR::Real_t>(*elem)) {
        asr_kind = ASR::down_cast<ASR::Real_t>(elem)->m_kind;
    } else if (ASR::is_a<ASR::Logical_t>(*elem)) {
        asr_kind = ASR::down_cast<ASR::Logical_t>(elem)->m_kind;
    }
    MLIR_TypeHandle i64_ty = MLIR_CreateTypeInteger(&ctx, 64, false);
    MLIR_AppendOpAttribute(&ctx, var_op, MLIR_CreateAttributeInteger(
        &ctx, str_lit("asr.type_kind"), asr_kind, i64_ty));
    if (is_array && array_len > 0) {
        MLIR_AppendOpAttribute(&ctx, var_op, MLIR_CreateAttributeInteger(
            &ctx, str_lit("asr.array_len"), array_len, i64_ty));
    }
}

void ASRToAsrDialectVisitor::attach_scope_regions(MLIR_OpHandle scope_op) {
    MLIR_OpHandle symtab = create_scope_container("asr.symtab", scope_symtab);
    MLIR_OpHandle metadata = create_scope_container("asr.metadata", scope_metadata);
    MLIR_OpHandle body = create_scope_container("asr.body", scope_body);
    for (MLIR_OpHandle var_op : scope_symtab) {
        ASR_ModuleStorageSetFieldOp(var_op, "parent_symtab", symtab);
    }
    if (symtab != MLIR_INVALID_HANDLE) {
        ASR_ModuleStorageSetFieldOp(scope_op, "symtab", symtab);
    }
    if (metadata != MLIR_INVALID_HANDLE) {
        ASR_ModuleStorageSetFieldOp(scope_op, "metadata", metadata);
    }
    if (body != MLIR_INVALID_HANDLE) {
        ASR_ModuleStorageSetFieldOp(scope_op, "body", body);
    }
}

void ASRToAsrDialectVisitor::append_scope_op(MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) {
        return;
    }
    switch (scope_region) {
        case ScopeRegion::Symtab:
            scope_symtab.push_back(op);
            break;
        case ScopeRegion::Metadata:
            scope_metadata.push_back(op);
            break;
        case ScopeRegion::Body:
            scope_body.push_back(op);
            break;
        case ScopeRegion::Module:
            append_module(op);
            break;
    }
}

void ASRToAsrDialectVisitor::append_current_stmt(MLIR_OpHandle op) {
    if (!suppress_module_append) {
        append_scope_op(op);
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

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_expr(ASR::expr_t &e) {
    last_value = MLIR_INVALID_HANDLE;
    ASR::BaseVisitor<ASRToAsrDialectVisitor>::visit_expr(e);
    return last_value;
}

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_stmt(ASR::stmt_t &s) {
    last_value = MLIR_INVALID_HANDLE;
    ASR::BaseVisitor<ASRToAsrDialectVisitor>::visit_stmt(s);
    return last_value;
}

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_expr_seq_value(ASR::expr_t **exprs, size_t n) {
    MLIR_ValueHandle last = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < n; ++i) {
        last = emit_expr(*exprs[i]);
    }
    return last;
}

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_stmt_seq_value(ASR::stmt_t **stmts, size_t n) {
    MLIR_ValueHandle last = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < n; ++i) {
        last = emit_stmt(*stmts[i]);
    }
    return last;
}

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_do_loop_head(const ASR::do_loop_head_t &h) {
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

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_array_index(const ASR::array_index_t &idx) {
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

MLIR_ValueHandle ASRToAsrDialectVisitor::emit_print_op(ASR::expr_t &text_expr) {
    MLIR_ValueHandle text = emit_expr(text_expr);
    last_value = ASR_CreatePrintOp(&ctx, default_loc(), text);
    return last_value;
}

void ASRToAsrDialectVisitor::emit_print_exprs(ASR::expr_t &text_expr) {
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

void ASRToAsrDialectVisitor::visit_Print(const ASR::Print_t &x) {
    if (!x.m_text) {
        throw AsrDialectError(
            "asr dialect: print with no format expression", x.base.base.loc);
    }
    emit_print_exprs(*x.m_text);
}

void ASRToAsrDialectVisitor::visit_StringFormat(const ASR::StringFormat_t &x) {
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
}

void ASRToAsrDialectVisitor::visit_FileWrite(const ASR::FileWrite_t &x) {
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

void ASRToAsrDialectVisitor::visit_DoLoop(const ASR::DoLoop_t &x) {
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
}

void ASRToAsrDialectVisitor::visit_Variable(const ASR::Variable_t &v) {
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
    string deps = emit_identifier_seq(v.m_dependencies, v.n_dependencies);
    last_value = ASR_CreateVariableOp(&ctx, default_loc(),
        MLIR_INVALID_HANDLE, asr_cstr(v.m_name), deps,
        (int64_t)v.m_intent, symbolic_value, value, (int64_t)v.m_storage,
        var_ty, type_decl, (int64_t)v.m_abi,
        (int64_t)v.m_access, (int64_t)v.m_presence, v.m_value_attr,
        v.m_target_attr, v.m_contiguous_attr,
        v.m_bindc_name ? asr_cstr(v.m_bindc_name) : str_lit(""),
        v.m_is_volatile, v.m_is_protected, (int64_t)v.m_pass_attr,
        v.m_self_argument ? asr_cstr(v.m_self_argument) : str_lit(""),
        nullptr, 0);
    attach_variable_type_attrs(last_value, *v.m_type);
    append_scope_op(last_value);
}

void ASRToAsrDialectVisitor::visit_ArrayConstructor(const ASR::ArrayConstructor_t &x) {
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
}

void ASRToAsrDialectVisitor::visit_ArrayItem(const ASR::ArrayItem_t &x) {
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
}

void ASRToAsrDialectVisitor::visit_ArrayConstant(const ASR::ArrayConstant_t &x) {
    if (!asr_al) {
        throw AsrDialectError("asr dialect: ASR allocator not set for ArrayConstant");
    }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    int64_t storage_format = (int64_t)x.m_storage_format;
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    // m_n_data is packed byte size (see asr_verify visit_ArrayConstant), not element count.
    size_t n_elems = ASRUtils::get_constant_ArrayConstant_size(
        const_cast<ASR::ArrayConstant_t *>(&x));
    std::vector<MLIR_OpHandle> elem_ops;
    elem_ops.reserve(n_elems);
    suppress_module_append = true;
    for (size_t i = 0; i < n_elems; ++i) {
        ASR::expr_t *elt = ASRUtils::fetch_ArrayConstant_value(*asr_al, x, (int)i);
        MLIR_OpHandle elem = emit_expr(*elt);
        if (elem != MLIR_INVALID_HANDLE) {
            elem_ops.push_back(elem);
        }
    }
    suppress_module_append = false;
    MLIR_OpHandle *buf = nullptr;
    size_t n = elem_ops.size();
    if (n > 0) {
        buf = (MLIR_OpHandle *)arena_alloc(arena, n * sizeof(MLIR_OpHandle));
        for (size_t i = 0; i < n; ++i) {
            buf[i] = elem_ops[i];
        }
    }
    last_value = ASR_CreateArrayConstantOp(&ctx, op_loc, buf, n, type, storage_format);
}

// >>> GENERATED VISITOR IMPLEMENTATIONS >>>
// Generated by asdl_to_asr_dialect.py — do not edit.

void ASRToAsrDialectVisitor::visit_ArrayBound(const ASR::ArrayBound_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    MLIR_ValueHandle dim = MLIR_INVALID_HANDLE;
        if (x.m_dim) { dim = emit_expr(*x.m_dim); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    int64_t bound = (int64_t)x.m_bound;
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArrayBoundOp(&ctx, op_loc, v, dim, type, bound, value);
}

void ASRToAsrDialectVisitor::visit_ArrayBroadcast(const ASR::ArrayBroadcast_t &x) {
    MLIR_ValueHandle array = emit_expr(*x.m_array);
    MLIR_ValueHandle shape = emit_expr(*x.m_shape);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArrayBroadcastOp(&ctx, op_loc, array, shape, type, value);
}

void ASRToAsrDialectVisitor::visit_ArrayIsContiguous(const ASR::ArrayIsContiguous_t &x) {
    MLIR_ValueHandle array = emit_expr(*x.m_array);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArrayIsContiguousOp(&ctx, op_loc, array, type, value);
}

void ASRToAsrDialectVisitor::visit_ArrayPack(const ASR::ArrayPack_t &x) {
    MLIR_ValueHandle array = emit_expr(*x.m_array);
    MLIR_ValueHandle mask = emit_expr(*x.m_mask);
    MLIR_ValueHandle vector = MLIR_INVALID_HANDLE;
        if (x.m_vector) { vector = emit_expr(*x.m_vector); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArrayPackOp(&ctx, op_loc, array, mask, vector, type, value);
}

void ASRToAsrDialectVisitor::visit_ArrayPhysicalCast(const ASR::ArrayPhysicalCast_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    int64_t old = (int64_t)x.m_old;
    int64_t new_ = (int64_t)x.m_new;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArrayPhysicalCastOp(&ctx, op_loc, arg, old, new_, type, value);
}

void ASRToAsrDialectVisitor::visit_ArrayRank(const ASR::ArrayRank_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArrayRankOp(&ctx, op_loc, v, type, value);
}

void ASRToAsrDialectVisitor::visit_ArrayReshape(const ASR::ArrayReshape_t &x) {
    MLIR_ValueHandle array = emit_expr(*x.m_array);
    MLIR_ValueHandle shape = emit_expr(*x.m_shape);
    MLIR_ValueHandle pad = MLIR_INVALID_HANDLE;
        if (x.m_pad) { pad = emit_expr(*x.m_pad); }
    MLIR_ValueHandle order = MLIR_INVALID_HANDLE;
        if (x.m_order) { order = emit_expr(*x.m_order); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArrayReshapeOp(&ctx, op_loc, array, shape, pad, order, type, value);
}

void ASRToAsrDialectVisitor::visit_ArraySection(const ASR::ArraySection_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_array_index_op_array(x.m_args, n_args);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArraySectionOp(&ctx, op_loc, v, args, n_args, type, value);
}

void ASRToAsrDialectVisitor::visit_ArraySize(const ASR::ArraySize_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    MLIR_ValueHandle dim = MLIR_INVALID_HANDLE;
        if (x.m_dim) { dim = emit_expr(*x.m_dim); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArraySizeOp(&ctx, op_loc, v, dim, type, value);
}

void ASRToAsrDialectVisitor::visit_ArrayTranspose(const ASR::ArrayTranspose_t &x) {
    MLIR_ValueHandle matrix = emit_expr(*x.m_matrix);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateArrayTransposeOp(&ctx, op_loc, matrix, type, value);
}

void ASRToAsrDialectVisitor::visit_BitCast(const ASR::BitCast_t &x) {
    MLIR_ValueHandle source = emit_expr(*x.m_source);
    MLIR_ValueHandle mold = emit_expr(*x.m_mold);
    MLIR_ValueHandle size = MLIR_INVALID_HANDLE;
        if (x.m_size) { size = emit_expr(*x.m_size); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateBitCastOp(&ctx, op_loc, source, mold, size, type, value);
}

void ASRToAsrDialectVisitor::visit_CLoc(const ASR::CLoc_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateCLocOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_CPtrCompare(const ASR::CPtrCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateCPtrCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_Cast(const ASR::Cast_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    int64_t kind = (int64_t)x.m_kind;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_ValueHandle dest = MLIR_INVALID_HANDLE;
        if (x.m_dest) { dest = emit_expr(*x.m_dest); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateCastOp(&ctx, op_loc, arg, kind, type, value, dest);
}

void ASRToAsrDialectVisitor::visit_CoarrayRef(const ASR::CoarrayRef_t &x) {
    MLIR_ValueHandle var = emit_expr(*x.m_var);
    size_t n_coindices = x.n_coindices;
        MLIR_OpHandle *coindices = emit_array_index_op_array(x.m_coindices, n_coindices);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateCoarrayRefOp(&ctx, op_loc, var, coindices, n_coindices, type, value);
}

void ASRToAsrDialectVisitor::visit_CompilerOptions(const ASR::CompilerOptions_t &x) {
    string compiler_options_str = asr_cstr(x.m_compiler_options_str);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateCompilerOptionsOp(&ctx, op_loc, compiler_options_str, type);
}

void ASRToAsrDialectVisitor::visit_ComplexBinOp(const ASR::ComplexBinOp_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateComplexBinOpOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_ComplexCompare(const ASR::ComplexCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateComplexCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_ComplexConstant(const ASR::ComplexConstant_t &x) {
    double re = x.m_re;
    double im = x.m_im;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateComplexConstantOp(&ctx, op_loc, re, im, type);
}

void ASRToAsrDialectVisitor::visit_ComplexConstructor(const ASR::ComplexConstructor_t &x) {
    MLIR_ValueHandle re = emit_expr(*x.m_re);
    MLIR_ValueHandle im = emit_expr(*x.m_im);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateComplexConstructorOp(&ctx, op_loc, re, im, type, value);
}

void ASRToAsrDialectVisitor::visit_ComplexIm(const ASR::ComplexIm_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateComplexImOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_ComplexRe(const ASR::ComplexRe_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateComplexReOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_ComplexUnaryMinus(const ASR::ComplexUnaryMinus_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateComplexUnaryMinusOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_DictConstant(const ASR::DictConstant_t &x) {
    size_t n_keys = x.n_keys;
        MLIR_OpHandle *keys = emit_expr_op_array(x.m_keys, n_keys);
    size_t n_values = x.n_values;
        MLIR_OpHandle *values = emit_expr_op_array(x.m_values, n_values);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDictConstantOp(&ctx, op_loc, keys, n_keys, values, n_values, type);
}

void ASRToAsrDialectVisitor::visit_DictContains(const ASR::DictContains_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDictContainsOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_DictItem(const ASR::DictItem_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle key = emit_expr(*x.m_key);
    MLIR_ValueHandle default_ = MLIR_INVALID_HANDLE;
        if (x.m_default) { default_ = emit_expr(*x.m_default); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDictItemOp(&ctx, op_loc, a, key, default_, type, value);
}

void ASRToAsrDialectVisitor::visit_DictLen(const ASR::DictLen_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDictLenOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_DictPop(const ASR::DictPop_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle key = emit_expr(*x.m_key);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDictPopOp(&ctx, op_loc, a, key, type, value);
}

void ASRToAsrDialectVisitor::visit_EnumConstructor(const ASR::EnumConstructor_t &x) {
    string dt_sym = emit_symbol_ref(*x.m_dt_sym);
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, n_args);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateEnumConstructorOp(&ctx, op_loc, dt_sym, args, n_args, type, value);
}

void ASRToAsrDialectVisitor::visit_EnumName(const ASR::EnumName_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    MLIR_TypeHandle enum_type = convert_type(*x.m_enum_type);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateEnumNameOp(&ctx, op_loc, v, enum_type, type, value);
}

void ASRToAsrDialectVisitor::visit_EnumStaticMember(const ASR::EnumStaticMember_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    string m = emit_symbol_ref(*x.m_m);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateEnumStaticMemberOp(&ctx, op_loc, v, m, type, value);
}

void ASRToAsrDialectVisitor::visit_EnumValue(const ASR::EnumValue_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    MLIR_TypeHandle enum_type = convert_type(*x.m_enum_type);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateEnumValueOp(&ctx, op_loc, v, enum_type, type, value);
}

void ASRToAsrDialectVisitor::visit_FunctionCall(const ASR::FunctionCall_t &x) {
    string name = emit_symbol_ref(*x.m_name);
    string original_name = str_lit("");
        if (x.m_original_name) { original_name = emit_symbol_ref(*x.m_original_name); }
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_product_op_array(x.m_args, n_args);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_ValueHandle dt = MLIR_INVALID_HANDLE;
        if (x.m_dt) { dt = emit_expr(*x.m_dt); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFunctionCallOp(&ctx, op_loc, name, original_name, args, n_args, type, value, dt);
}

void ASRToAsrDialectVisitor::visit_FunctionParam(const ASR::FunctionParam_t &x) {
    int64_t param_number = x.m_param_number;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFunctionParamOp(&ctx, op_loc, param_number, type, value);
}

void ASRToAsrDialectVisitor::visit_GetPointer(const ASR::GetPointer_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateGetPointerOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_GpuBlockIndex(const ASR::GpuBlockIndex_t &x) {
    int64_t dim = x.m_dim;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateGpuBlockIndexOp(&ctx, op_loc, dim, type, value);
}

void ASRToAsrDialectVisitor::visit_GpuBlockSize(const ASR::GpuBlockSize_t &x) {
    int64_t dim = x.m_dim;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateGpuBlockSizeOp(&ctx, op_loc, dim, type, value);
}

void ASRToAsrDialectVisitor::visit_GpuThreadIndex(const ASR::GpuThreadIndex_t &x) {
    int64_t dim = x.m_dim;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateGpuThreadIndexOp(&ctx, op_loc, dim, type, value);
}

void ASRToAsrDialectVisitor::visit_Iachar(const ASR::Iachar_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIacharOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_Ichar(const ASR::Ichar_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIcharOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_IfExp(const ASR::IfExp_t &x) {
    MLIR_ValueHandle test = emit_expr(*x.m_test);
    MLIR_ValueHandle body = emit_expr(*x.m_body);
    MLIR_ValueHandle orelse = emit_expr(*x.m_orelse);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIfExpOp(&ctx, op_loc, test, body, orelse, type, value);
}

void ASRToAsrDialectVisitor::visit_ImpliedDoLoop(const ASR::ImpliedDoLoop_t &x) {
    size_t n_values = x.n_values;
        MLIR_OpHandle *values = emit_expr_op_array(x.m_values, n_values);
    MLIR_ValueHandle var = emit_expr(*x.m_var);
    MLIR_ValueHandle start = emit_expr(*x.m_start);
    MLIR_ValueHandle end = emit_expr(*x.m_end);
    MLIR_ValueHandle increment = MLIR_INVALID_HANDLE;
        if (x.m_increment) { increment = emit_expr(*x.m_increment); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateImpliedDoLoopOp(&ctx, op_loc, values, n_values, var, start, end, increment, type, value);
}

void ASRToAsrDialectVisitor::visit_IntegerBinOp(const ASR::IntegerBinOp_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntegerBinOpOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_IntegerBitLen(const ASR::IntegerBitLen_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntegerBitLenOp(&ctx, op_loc, a, type, value);
}

void ASRToAsrDialectVisitor::visit_IntegerBitNot(const ASR::IntegerBitNot_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntegerBitNotOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_IntegerCompare(const ASR::IntegerCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntegerCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_IntegerConstant(const ASR::IntegerConstant_t &x) {
    int64_t n = x.m_n;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    int64_t intboz_type = (int64_t)x.m_intboz_type;
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntegerConstantOp(&ctx, op_loc, n, type, intboz_type);
}

void ASRToAsrDialectVisitor::visit_IntegerUnaryMinus(const ASR::IntegerUnaryMinus_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntegerUnaryMinusOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_IntrinsicArrayFunction(const ASR::IntrinsicArrayFunction_t &x) {
    int64_t arr_intrinsic_id = x.m_arr_intrinsic_id;
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, n_args);
    int64_t overload_id = x.m_overload_id;
    MLIR_TypeHandle type = MLIR_INVALID_HANDLE;
        if (x.m_type) { type = convert_type(*x.m_type); }
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntrinsicArrayFunctionOp(&ctx, op_loc, arr_intrinsic_id, args, n_args, overload_id, type, value);
}

void ASRToAsrDialectVisitor::visit_IntrinsicElementalFunction(const ASR::IntrinsicElementalFunction_t &x) {
    int64_t intrinsic_id = x.m_intrinsic_id;
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, n_args);
    int64_t overload_id = x.m_overload_id;
    MLIR_TypeHandle type = MLIR_INVALID_HANDLE;
        if (x.m_type) { type = convert_type(*x.m_type); }
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntrinsicElementalFunctionOp(&ctx, op_loc, intrinsic_id, args, n_args, overload_id, type, value);
}

void ASRToAsrDialectVisitor::visit_IntrinsicImpureFunction(const ASR::IntrinsicImpureFunction_t &x) {
    int64_t impure_intrinsic_id = x.m_impure_intrinsic_id;
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, n_args);
    int64_t overload_id = x.m_overload_id;
    MLIR_TypeHandle type = MLIR_INVALID_HANDLE;
        if (x.m_type) { type = convert_type(*x.m_type); }
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntrinsicImpureFunctionOp(&ctx, op_loc, impure_intrinsic_id, args, n_args, overload_id, type, value);
}

void ASRToAsrDialectVisitor::visit_ListCompare(const ASR::ListCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_ListConcat(const ASR::ListConcat_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListConcatOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_ListConstant(const ASR::ListConstant_t &x) {
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, n_args);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListConstantOp(&ctx, op_loc, args, n_args, type);
}

void ASRToAsrDialectVisitor::visit_ListContains(const ASR::ListContains_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListContainsOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_ListCount(const ASR::ListCount_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_ValueHandle ele = emit_expr(*x.m_ele);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListCountOp(&ctx, op_loc, arg, ele, type, value);
}

void ASRToAsrDialectVisitor::visit_ListItem(const ASR::ListItem_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle pos = emit_expr(*x.m_pos);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListItemOp(&ctx, op_loc, a, pos, type, value);
}

void ASRToAsrDialectVisitor::visit_ListLen(const ASR::ListLen_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListLenOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_ListRepeat(const ASR::ListRepeat_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListRepeatOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_ListSection(const ASR::ListSection_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle section = emit_array_index(x.m_section);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListSectionOp(&ctx, op_loc, a, section, type, value);
}

void ASRToAsrDialectVisitor::visit_LogicalBinOp(const ASR::LogicalBinOp_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateLogicalBinOpOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_LogicalCompare(const ASR::LogicalCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateLogicalCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_LogicalConstant(const ASR::LogicalConstant_t &x) {
    bool value = x.m_value;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateLogicalConstantOp(&ctx, op_loc, value, type);
}

void ASRToAsrDialectVisitor::visit_LogicalNot(const ASR::LogicalNot_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateLogicalNotOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_NamedExpr(const ASR::NamedExpr_t &x) {
    MLIR_ValueHandle target = emit_expr(*x.m_target);
    MLIR_ValueHandle value = emit_expr(*x.m_value);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateNamedExprOp(&ctx, op_loc, target, value, type);
}

void ASRToAsrDialectVisitor::visit_OverloadedBinOp(const ASR::OverloadedBinOp_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_ValueHandle overloaded = emit_expr(*x.m_overloaded);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateOverloadedBinOpOp(&ctx, op_loc, left, op, right, type, value, overloaded);
}

void ASRToAsrDialectVisitor::visit_OverloadedBoolOp(const ASR::OverloadedBoolOp_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_ValueHandle overloaded = emit_expr(*x.m_overloaded);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateOverloadedBoolOpOp(&ctx, op_loc, left, op, right, type, value, overloaded);
}

void ASRToAsrDialectVisitor::visit_OverloadedCompare(const ASR::OverloadedCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_ValueHandle overloaded = emit_expr(*x.m_overloaded);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateOverloadedCompareOp(&ctx, op_loc, left, op, right, type, value, overloaded);
}

void ASRToAsrDialectVisitor::visit_OverloadedStringConcat(const ASR::OverloadedStringConcat_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_ValueHandle overloaded = emit_expr(*x.m_overloaded);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateOverloadedStringConcatOp(&ctx, op_loc, left, right, type, value, overloaded);
}

void ASRToAsrDialectVisitor::visit_OverloadedUnaryMinus(const ASR::OverloadedUnaryMinus_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_ValueHandle overloaded = emit_expr(*x.m_overloaded);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateOverloadedUnaryMinusOp(&ctx, op_loc, arg, type, value, overloaded);
}

void ASRToAsrDialectVisitor::visit_PointerAssociated(const ASR::PointerAssociated_t &x) {
    MLIR_ValueHandle ptr = emit_expr(*x.m_ptr);
    MLIR_ValueHandle tgt = MLIR_INVALID_HANDLE;
        if (x.m_tgt) { tgt = emit_expr(*x.m_tgt); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreatePointerAssociatedOp(&ctx, op_loc, ptr, tgt, type, value);
}

void ASRToAsrDialectVisitor::visit_PointerNullConstant(const ASR::PointerNullConstant_t &x) {
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle var_expr = MLIR_INVALID_HANDLE;
        if (x.m_var_expr) { var_expr = emit_expr(*x.m_var_expr); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreatePointerNullConstantOp(&ctx, op_loc, type, var_expr);
}

void ASRToAsrDialectVisitor::visit_PointerToCPtr(const ASR::PointerToCPtr_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreatePointerToCPtrOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_RealBinOp(const ASR::RealBinOp_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateRealBinOpOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_RealCompare(const ASR::RealCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateRealCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_RealConstant(const ASR::RealConstant_t &x) {
    double r = x.m_r;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateRealConstantOp(&ctx, op_loc, r, type);
}

void ASRToAsrDialectVisitor::visit_RealCopySign(const ASR::RealCopySign_t &x) {
    MLIR_ValueHandle target = emit_expr(*x.m_target);
    MLIR_ValueHandle source = emit_expr(*x.m_source);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateRealCopySignOp(&ctx, op_loc, target, source, type, value);
}

void ASRToAsrDialectVisitor::visit_RealSqrt(const ASR::RealSqrt_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateRealSqrtOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_RealUnaryMinus(const ASR::RealUnaryMinus_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateRealUnaryMinusOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_SetConstant(const ASR::SetConstant_t &x) {
    size_t n_elements = x.n_elements;
        MLIR_OpHandle *elements = emit_expr_op_array(x.m_elements, n_elements);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSetConstantOp(&ctx, op_loc, elements, n_elements, type);
}

void ASRToAsrDialectVisitor::visit_SetContains(const ASR::SetContains_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSetContainsOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_SetLen(const ASR::SetLen_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSetLenOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_SetPop(const ASR::SetPop_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSetPopOp(&ctx, op_loc, a, type, value);
}

void ASRToAsrDialectVisitor::visit_SizeOfType(const ASR::SizeOfType_t &x) {
    MLIR_TypeHandle arg = convert_type(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSizeOfTypeOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_StringChr(const ASR::StringChr_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringChrOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_StringCompare(const ASR::StringCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_StringConcat(const ASR::StringConcat_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringConcatOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_StringConstant(const ASR::StringConstant_t &x) {
    string s = asr_cstr(x.m_s);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringConstantOp(&ctx, op_loc, s, type);
}

void ASRToAsrDialectVisitor::visit_StringContains(const ASR::StringContains_t &x) {
    MLIR_ValueHandle substr = emit_expr(*x.m_substr);
    MLIR_ValueHandle str = emit_expr(*x.m_str);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringContainsOp(&ctx, op_loc, substr, str, type, value);
}

void ASRToAsrDialectVisitor::visit_StringItem(const ASR::StringItem_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_ValueHandle idx = emit_expr(*x.m_idx);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringItemOp(&ctx, op_loc, arg, idx, type, value);
}

void ASRToAsrDialectVisitor::visit_StringLen(const ASR::StringLen_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringLenOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_StringOrd(const ASR::StringOrd_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringOrdOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_StringPhysicalCast(const ASR::StringPhysicalCast_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    int64_t old = (int64_t)x.m_old;
    int64_t new_ = (int64_t)x.m_new;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringPhysicalCastOp(&ctx, op_loc, arg, old, new_, type, value);
}

void ASRToAsrDialectVisitor::visit_StringRepeat(const ASR::StringRepeat_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringRepeatOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_StringSection(const ASR::StringSection_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_ValueHandle start = MLIR_INVALID_HANDLE;
        if (x.m_start) { start = emit_expr(*x.m_start); }
    MLIR_ValueHandle end = MLIR_INVALID_HANDLE;
        if (x.m_end) { end = emit_expr(*x.m_end); }
    MLIR_ValueHandle step = MLIR_INVALID_HANDLE;
        if (x.m_step) { step = emit_expr(*x.m_step); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStringSectionOp(&ctx, op_loc, arg, start, end, step, type, value);
}

void ASRToAsrDialectVisitor::visit_StructConstant(const ASR::StructConstant_t &x) {
    string dt_sym = emit_symbol_ref(*x.m_dt_sym);
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_product_op_array(x.m_args, n_args);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStructConstantOp(&ctx, op_loc, dt_sym, args, n_args, type);
}

void ASRToAsrDialectVisitor::visit_StructConstructor(const ASR::StructConstructor_t &x) {
    string dt_sym = emit_symbol_ref(*x.m_dt_sym);
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_product_op_array(x.m_args, n_args);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStructConstructorOp(&ctx, op_loc, dt_sym, args, n_args, type, value);
}

void ASRToAsrDialectVisitor::visit_StructInstanceMember(const ASR::StructInstanceMember_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    string m = emit_symbol_ref(*x.m_m);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStructInstanceMemberOp(&ctx, op_loc, v, m, type, value);
}

void ASRToAsrDialectVisitor::visit_StructStaticMember(const ASR::StructStaticMember_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    string m = emit_symbol_ref(*x.m_m);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStructStaticMemberOp(&ctx, op_loc, v, m, type, value);
}

void ASRToAsrDialectVisitor::visit_SymbolicCompare(const ASR::SymbolicCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSymbolicCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_TupleCompare(const ASR::TupleCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateTupleCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_TupleConcat(const ASR::TupleConcat_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateTupleConcatOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_TupleConstant(const ASR::TupleConstant_t &x) {
    size_t n_elements = x.n_elements;
        MLIR_OpHandle *elements = emit_expr_op_array(x.m_elements, n_elements);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateTupleConstantOp(&ctx, op_loc, elements, n_elements, type);
}

void ASRToAsrDialectVisitor::visit_TupleContains(const ASR::TupleContains_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateTupleContainsOp(&ctx, op_loc, left, right, type, value);
}

void ASRToAsrDialectVisitor::visit_TupleItem(const ASR::TupleItem_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle pos = emit_expr(*x.m_pos);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateTupleItemOp(&ctx, op_loc, a, pos, type, value);
}

void ASRToAsrDialectVisitor::visit_TupleLen(const ASR::TupleLen_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = emit_expr(*x.m_value);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateTupleLenOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_TypeInquiry(const ASR::TypeInquiry_t &x) {
    int64_t inquiry_id = x.m_inquiry_id;
    MLIR_TypeHandle arg_type = convert_type(*x.m_arg_type);
    MLIR_ValueHandle arg = MLIR_INVALID_HANDLE;
        if (x.m_arg) { arg = emit_expr(*x.m_arg); }
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = emit_expr(*x.m_value);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateTypeInquiryOp(&ctx, op_loc, inquiry_id, arg_type, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_UnionConstructor(const ASR::UnionConstructor_t &x) {
    string dt_sym = emit_symbol_ref(*x.m_dt_sym);
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, n_args);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateUnionConstructorOp(&ctx, op_loc, dt_sym, args, n_args, type, value);
}

void ASRToAsrDialectVisitor::visit_UnionInstanceMember(const ASR::UnionInstanceMember_t &x) {
    MLIR_ValueHandle v = emit_expr(*x.m_v);
    string m = emit_symbol_ref(*x.m_m);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateUnionInstanceMemberOp(&ctx, op_loc, v, m, type, value);
}

void ASRToAsrDialectVisitor::visit_UnsignedIntegerBinOp(const ASR::UnsignedIntegerBinOp_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateUnsignedIntegerBinOpOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_UnsignedIntegerBitNot(const ASR::UnsignedIntegerBitNot_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateUnsignedIntegerBitNotOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_UnsignedIntegerCompare(const ASR::UnsignedIntegerCompare_t &x) {
    MLIR_ValueHandle left = emit_expr(*x.m_left);
    int64_t op = (int64_t)x.m_op;
    MLIR_ValueHandle right = emit_expr(*x.m_right);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateUnsignedIntegerCompareOp(&ctx, op_loc, left, op, right, type, value);
}

void ASRToAsrDialectVisitor::visit_UnsignedIntegerConstant(const ASR::UnsignedIntegerConstant_t &x) {
    int64_t n = x.m_n;
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateUnsignedIntegerConstantOp(&ctx, op_loc, n, type);
}

void ASRToAsrDialectVisitor::visit_UnsignedIntegerUnaryMinus(const ASR::UnsignedIntegerUnaryMinus_t &x) {
    MLIR_ValueHandle arg = emit_expr(*x.m_arg);
    MLIR_TypeHandle type = convert_type(*x.m_type);
    MLIR_ValueHandle value = MLIR_INVALID_HANDLE;
        if (x.m_value) { value = emit_expr(*x.m_value); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateUnsignedIntegerUnaryMinusOp(&ctx, op_loc, arg, type, value);
}

void ASRToAsrDialectVisitor::visit_Var(const ASR::Var_t &x) {
    string v = emit_symbol_ref(*x.m_v);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateVarOp(&ctx, op_loc, v);
}

void ASRToAsrDialectVisitor::visit_Allocate(const ASR::Allocate_t &x) {
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_product_op_array(x.m_args, n_args);
    MLIR_ValueHandle stat = MLIR_INVALID_HANDLE;
        if (x.m_stat) { stat = emit_expr(*x.m_stat); }
    MLIR_ValueHandle errmsg = MLIR_INVALID_HANDLE;
        if (x.m_errmsg) { errmsg = emit_expr(*x.m_errmsg); }
    MLIR_ValueHandle source = MLIR_INVALID_HANDLE;
        if (x.m_source) { source = emit_expr(*x.m_source); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateAllocateOp(&ctx, op_loc, args, n_args, stat, errmsg, source);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Assert(const ASR::Assert_t &x) {
    MLIR_ValueHandle test = emit_expr(*x.m_test);
    MLIR_ValueHandle msg = MLIR_INVALID_HANDLE;
        if (x.m_msg) { msg = emit_expr(*x.m_msg); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateAssertOp(&ctx, op_loc, test, msg);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Assign(const ASR::Assign_t &x) {
    int64_t label = x.m_label;
    string variable = asr_cstr(x.m_variable);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateAssignOp(&ctx, op_loc, label, variable);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Assignment(const ASR::Assignment_t &x) {
    MLIR_ValueHandle target = emit_expr(*x.m_target);
    MLIR_ValueHandle value = emit_expr(*x.m_value);
    MLIR_ValueHandle overloaded = MLIR_INVALID_HANDLE;
        if (x.m_overloaded) { overloaded = emit_stmt(*x.m_overloaded); }
    bool realloc_lhs = x.m_realloc_lhs;
    bool move_allocation = x.m_move_allocation;
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateAssignmentOp(&ctx, op_loc, target, value, overloaded, realloc_lhs, move_allocation);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Associate(const ASR::Associate_t &x) {
    MLIR_ValueHandle target = emit_expr(*x.m_target);
    MLIR_ValueHandle value = emit_expr(*x.m_value);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateAssociateOp(&ctx, op_loc, target, value);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_AssociateBlockCall(const ASR::AssociateBlockCall_t &x) {
    string m = emit_symbol_ref(*x.m_m);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateAssociateBlockCallOp(&ctx, op_loc, m);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_BlockCall(const ASR::BlockCall_t &x) {
    int64_t label = x.m_label;
    string m = emit_symbol_ref(*x.m_m);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateBlockCallOp(&ctx, op_loc, label, m);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_CPtrToPointer(const ASR::CPtrToPointer_t &x) {
    MLIR_ValueHandle cptr = emit_expr(*x.m_cptr);
    MLIR_ValueHandle ptr = emit_expr(*x.m_ptr);
    MLIR_ValueHandle shape = MLIR_INVALID_HANDLE;
        if (x.m_shape) { shape = emit_expr(*x.m_shape); }
    MLIR_ValueHandle lower_bounds = MLIR_INVALID_HANDLE;
        if (x.m_lower_bounds) { lower_bounds = emit_expr(*x.m_lower_bounds); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateCPtrToPointerOp(&ctx, op_loc, cptr, ptr, shape, lower_bounds);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Cycle(const ASR::Cycle_t &x) {
    string stmt_name = asr_cstr(x.m_stmt_name);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateCycleOp(&ctx, op_loc, stmt_name);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_DebugCheckArrayBounds(const ASR::DebugCheckArrayBounds_t &x) {
    MLIR_ValueHandle target = emit_expr(*x.m_target);
    size_t n_components = x.n_components;
        MLIR_OpHandle *components = emit_expr_op_array(x.m_components, n_components);
    bool move_allocation = x.m_move_allocation;
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDebugCheckArrayBoundsOp(&ctx, op_loc, target, components, n_components, move_allocation);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_DictClear(const ASR::DictClear_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDictClearOp(&ctx, op_loc, a);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_DictInsert(const ASR::DictInsert_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle key = emit_expr(*x.m_key);
    MLIR_ValueHandle value = emit_expr(*x.m_value);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDictInsertOp(&ctx, op_loc, a, key, value);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_DoConcurrentLoop(const ASR::DoConcurrentLoop_t &x) {
    size_t n_head = x.n_head;
        MLIR_OpHandle *head = emit_product_op_array(x.m_head, n_head);
    size_t n_shared = x.n_shared;
        MLIR_OpHandle *shared = emit_expr_op_array(x.m_shared, n_shared);
    size_t n_local = x.n_local;
        MLIR_OpHandle *local = emit_expr_op_array(x.m_local, n_local);
    size_t n_reduction = x.n_reduction;
        MLIR_OpHandle *reduction = emit_product_op_array(x.m_reduction, n_reduction);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_stmt_op_array(x.m_body, n_body);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateDoConcurrentLoopOp(&ctx, op_loc, head, n_head, shared, n_shared, local, n_local, reduction, n_reduction, body, n_body);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ErrorStop(const ASR::ErrorStop_t &x) {
    MLIR_ValueHandle code = MLIR_INVALID_HANDLE;
        if (x.m_code) { code = emit_expr(*x.m_code); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateErrorStopOp(&ctx, op_loc, code);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Exit(const ASR::Exit_t &x) {
    string stmt_name = asr_cstr(x.m_stmt_name);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateExitOp(&ctx, op_loc, stmt_name);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ExplicitDeallocate(const ASR::ExplicitDeallocate_t &x) {
    size_t n_vars = x.n_vars;
        MLIR_OpHandle *vars = emit_expr_op_array(x.m_vars, n_vars);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateExplicitDeallocateOp(&ctx, op_loc, vars, n_vars);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Expr(const ASR::Expr_t &x) {
    MLIR_ValueHandle expression = emit_expr(*x.m_expression);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateExprOp(&ctx, op_loc, expression);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_FileBackspace(const ASR::FileBackspace_t &x) {
    int64_t label = x.m_label;
    MLIR_ValueHandle unit = MLIR_INVALID_HANDLE;
        if (x.m_unit) { unit = emit_expr(*x.m_unit); }
    MLIR_ValueHandle iostat = MLIR_INVALID_HANDLE;
        if (x.m_iostat) { iostat = emit_expr(*x.m_iostat); }
    MLIR_ValueHandle err = MLIR_INVALID_HANDLE;
        if (x.m_err) { err = emit_expr(*x.m_err); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFileBackspaceOp(&ctx, op_loc, label, unit, iostat, err);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_FileClose(const ASR::FileClose_t &x) {
    int64_t label = x.m_label;
    MLIR_ValueHandle unit = MLIR_INVALID_HANDLE;
        if (x.m_unit) { unit = emit_expr(*x.m_unit); }
    MLIR_ValueHandle iostat = MLIR_INVALID_HANDLE;
        if (x.m_iostat) { iostat = emit_expr(*x.m_iostat); }
    MLIR_ValueHandle iomsg = MLIR_INVALID_HANDLE;
        if (x.m_iomsg) { iomsg = emit_expr(*x.m_iomsg); }
    MLIR_ValueHandle err = MLIR_INVALID_HANDLE;
        if (x.m_err) { err = emit_expr(*x.m_err); }
    MLIR_ValueHandle status = MLIR_INVALID_HANDLE;
        if (x.m_status) { status = emit_expr(*x.m_status); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFileCloseOp(&ctx, op_loc, label, unit, iostat, iomsg, err, status);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_FileEndfile(const ASR::FileEndfile_t &x) {
    int64_t label = x.m_label;
    MLIR_ValueHandle unit = MLIR_INVALID_HANDLE;
        if (x.m_unit) { unit = emit_expr(*x.m_unit); }
    MLIR_ValueHandle iostat = MLIR_INVALID_HANDLE;
        if (x.m_iostat) { iostat = emit_expr(*x.m_iostat); }
    MLIR_ValueHandle err = MLIR_INVALID_HANDLE;
        if (x.m_err) { err = emit_expr(*x.m_err); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFileEndfileOp(&ctx, op_loc, label, unit, iostat, err);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_FileInquire(const ASR::FileInquire_t &x) {
    int64_t label = x.m_label;
    MLIR_ValueHandle unit = MLIR_INVALID_HANDLE;
        if (x.m_unit) { unit = emit_expr(*x.m_unit); }
    MLIR_ValueHandle file = MLIR_INVALID_HANDLE;
        if (x.m_file) { file = emit_expr(*x.m_file); }
    MLIR_ValueHandle iostat = MLIR_INVALID_HANDLE;
        if (x.m_iostat) { iostat = emit_expr(*x.m_iostat); }
    MLIR_ValueHandle err = MLIR_INVALID_HANDLE;
        if (x.m_err) { err = emit_expr(*x.m_err); }
    MLIR_ValueHandle exist = MLIR_INVALID_HANDLE;
        if (x.m_exist) { exist = emit_expr(*x.m_exist); }
    MLIR_ValueHandle opened = MLIR_INVALID_HANDLE;
        if (x.m_opened) { opened = emit_expr(*x.m_opened); }
    MLIR_ValueHandle number = MLIR_INVALID_HANDLE;
        if (x.m_number) { number = emit_expr(*x.m_number); }
    MLIR_ValueHandle named = MLIR_INVALID_HANDLE;
        if (x.m_named) { named = emit_expr(*x.m_named); }
    MLIR_ValueHandle name = MLIR_INVALID_HANDLE;
        if (x.m_name) { name = emit_expr(*x.m_name); }
    MLIR_ValueHandle access = MLIR_INVALID_HANDLE;
        if (x.m_access) { access = emit_expr(*x.m_access); }
    MLIR_ValueHandle sequential = MLIR_INVALID_HANDLE;
        if (x.m_sequential) { sequential = emit_expr(*x.m_sequential); }
    MLIR_ValueHandle direct = MLIR_INVALID_HANDLE;
        if (x.m_direct) { direct = emit_expr(*x.m_direct); }
    MLIR_ValueHandle form = MLIR_INVALID_HANDLE;
        if (x.m_form) { form = emit_expr(*x.m_form); }
    MLIR_ValueHandle formatted = MLIR_INVALID_HANDLE;
        if (x.m_formatted) { formatted = emit_expr(*x.m_formatted); }
    MLIR_ValueHandle unformatted = MLIR_INVALID_HANDLE;
        if (x.m_unformatted) { unformatted = emit_expr(*x.m_unformatted); }
    MLIR_ValueHandle recl = MLIR_INVALID_HANDLE;
        if (x.m_recl) { recl = emit_expr(*x.m_recl); }
    MLIR_ValueHandle nextrec = MLIR_INVALID_HANDLE;
        if (x.m_nextrec) { nextrec = emit_expr(*x.m_nextrec); }
    MLIR_ValueHandle blank = MLIR_INVALID_HANDLE;
        if (x.m_blank) { blank = emit_expr(*x.m_blank); }
    MLIR_ValueHandle position = MLIR_INVALID_HANDLE;
        if (x.m_position) { position = emit_expr(*x.m_position); }
    MLIR_ValueHandle action = MLIR_INVALID_HANDLE;
        if (x.m_action) { action = emit_expr(*x.m_action); }
    MLIR_ValueHandle read_ = MLIR_INVALID_HANDLE;
        if (x.m_read) { read_ = emit_expr(*x.m_read); }
    MLIR_ValueHandle write_ = MLIR_INVALID_HANDLE;
        if (x.m_write) { write_ = emit_expr(*x.m_write); }
    MLIR_ValueHandle readwrite = MLIR_INVALID_HANDLE;
        if (x.m_readwrite) { readwrite = emit_expr(*x.m_readwrite); }
    MLIR_ValueHandle delim = MLIR_INVALID_HANDLE;
        if (x.m_delim) { delim = emit_expr(*x.m_delim); }
    MLIR_ValueHandle pad = MLIR_INVALID_HANDLE;
        if (x.m_pad) { pad = emit_expr(*x.m_pad); }
    MLIR_ValueHandle flen = MLIR_INVALID_HANDLE;
        if (x.m_flen) { flen = emit_expr(*x.m_flen); }
    MLIR_ValueHandle blocksize = MLIR_INVALID_HANDLE;
        if (x.m_blocksize) { blocksize = emit_expr(*x.m_blocksize); }
    MLIR_ValueHandle convert = MLIR_INVALID_HANDLE;
        if (x.m_convert) { convert = emit_expr(*x.m_convert); }
    MLIR_ValueHandle carriagecontrol = MLIR_INVALID_HANDLE;
        if (x.m_carriagecontrol) { carriagecontrol = emit_expr(*x.m_carriagecontrol); }
    MLIR_ValueHandle size = MLIR_INVALID_HANDLE;
        if (x.m_size) { size = emit_expr(*x.m_size); }
    MLIR_ValueHandle pos = MLIR_INVALID_HANDLE;
        if (x.m_pos) { pos = emit_expr(*x.m_pos); }
    MLIR_ValueHandle iolength = MLIR_INVALID_HANDLE;
        if (x.m_iolength) { iolength = emit_expr(*x.m_iolength); }
    size_t n_iolength_vars = x.n_iolength_vars;
        MLIR_OpHandle *iolength_vars = emit_expr_op_array(x.m_iolength_vars, n_iolength_vars);
    MLIR_ValueHandle decimal = MLIR_INVALID_HANDLE;
        if (x.m_decimal) { decimal = emit_expr(*x.m_decimal); }
    MLIR_ValueHandle sign = MLIR_INVALID_HANDLE;
        if (x.m_sign) { sign = emit_expr(*x.m_sign); }
    MLIR_ValueHandle encoding = MLIR_INVALID_HANDLE;
        if (x.m_encoding) { encoding = emit_expr(*x.m_encoding); }
    MLIR_ValueHandle stream = MLIR_INVALID_HANDLE;
        if (x.m_stream) { stream = emit_expr(*x.m_stream); }
    MLIR_ValueHandle iomsg = MLIR_INVALID_HANDLE;
        if (x.m_iomsg) { iomsg = emit_expr(*x.m_iomsg); }
    MLIR_ValueHandle round = MLIR_INVALID_HANDLE;
        if (x.m_round) { round = emit_expr(*x.m_round); }
    MLIR_ValueHandle pending = MLIR_INVALID_HANDLE;
        if (x.m_pending) { pending = emit_expr(*x.m_pending); }
    MLIR_ValueHandle asynchronous = MLIR_INVALID_HANDLE;
        if (x.m_asynchronous) { asynchronous = emit_expr(*x.m_asynchronous); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFileInquireOp(&ctx, op_loc, label, unit, file, iostat, err, exist, opened, number, named, name, access, sequential, direct, form, formatted, unformatted, recl, nextrec, blank, position, action, read_, write_, readwrite, delim, pad, flen, blocksize, convert, carriagecontrol, size, pos, iolength, iolength_vars, n_iolength_vars, decimal, sign, encoding, stream, iomsg, round, pending, asynchronous);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_FileOpen(const ASR::FileOpen_t &x) {
    int64_t label = x.m_label;
    MLIR_ValueHandle newunit = MLIR_INVALID_HANDLE;
        if (x.m_newunit) { newunit = emit_expr(*x.m_newunit); }
    MLIR_ValueHandle filename = MLIR_INVALID_HANDLE;
        if (x.m_filename) { filename = emit_expr(*x.m_filename); }
    MLIR_ValueHandle status = MLIR_INVALID_HANDLE;
        if (x.m_status) { status = emit_expr(*x.m_status); }
    MLIR_ValueHandle form = MLIR_INVALID_HANDLE;
        if (x.m_form) { form = emit_expr(*x.m_form); }
    MLIR_ValueHandle access = MLIR_INVALID_HANDLE;
        if (x.m_access) { access = emit_expr(*x.m_access); }
    MLIR_ValueHandle iostat = MLIR_INVALID_HANDLE;
        if (x.m_iostat) { iostat = emit_expr(*x.m_iostat); }
    MLIR_ValueHandle iomsg = MLIR_INVALID_HANDLE;
        if (x.m_iomsg) { iomsg = emit_expr(*x.m_iomsg); }
    MLIR_ValueHandle action = MLIR_INVALID_HANDLE;
        if (x.m_action) { action = emit_expr(*x.m_action); }
    MLIR_ValueHandle delim = MLIR_INVALID_HANDLE;
        if (x.m_delim) { delim = emit_expr(*x.m_delim); }
    MLIR_ValueHandle recl = MLIR_INVALID_HANDLE;
        if (x.m_recl) { recl = emit_expr(*x.m_recl); }
    MLIR_ValueHandle position = MLIR_INVALID_HANDLE;
        if (x.m_position) { position = emit_expr(*x.m_position); }
    MLIR_ValueHandle blank = MLIR_INVALID_HANDLE;
        if (x.m_blank) { blank = emit_expr(*x.m_blank); }
    MLIR_ValueHandle encoding = MLIR_INVALID_HANDLE;
        if (x.m_encoding) { encoding = emit_expr(*x.m_encoding); }
    MLIR_ValueHandle sign = MLIR_INVALID_HANDLE;
        if (x.m_sign) { sign = emit_expr(*x.m_sign); }
    MLIR_ValueHandle decimal = MLIR_INVALID_HANDLE;
        if (x.m_decimal) { decimal = emit_expr(*x.m_decimal); }
    MLIR_ValueHandle round = MLIR_INVALID_HANDLE;
        if (x.m_round) { round = emit_expr(*x.m_round); }
    MLIR_ValueHandle pad = MLIR_INVALID_HANDLE;
        if (x.m_pad) { pad = emit_expr(*x.m_pad); }
    MLIR_ValueHandle asynchronous = MLIR_INVALID_HANDLE;
        if (x.m_asynchronous) { asynchronous = emit_expr(*x.m_asynchronous); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFileOpenOp(&ctx, op_loc, label, newunit, filename, status, form, access, iostat, iomsg, action, delim, recl, position, blank, encoding, sign, decimal, round, pad, asynchronous);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_FileRead(const ASR::FileRead_t &x) {
    int64_t label = x.m_label;
    MLIR_ValueHandle unit = MLIR_INVALID_HANDLE;
        if (x.m_unit) { unit = emit_expr(*x.m_unit); }
    MLIR_ValueHandle fmt = MLIR_INVALID_HANDLE;
        if (x.m_fmt) { fmt = emit_expr(*x.m_fmt); }
    MLIR_ValueHandle iomsg = MLIR_INVALID_HANDLE;
        if (x.m_iomsg) { iomsg = emit_expr(*x.m_iomsg); }
    MLIR_ValueHandle iostat = MLIR_INVALID_HANDLE;
        if (x.m_iostat) { iostat = emit_expr(*x.m_iostat); }
    MLIR_ValueHandle advance = MLIR_INVALID_HANDLE;
        if (x.m_advance) { advance = emit_expr(*x.m_advance); }
    MLIR_ValueHandle size = MLIR_INVALID_HANDLE;
        if (x.m_size) { size = emit_expr(*x.m_size); }
    MLIR_ValueHandle id = MLIR_INVALID_HANDLE;
        if (x.m_id) { id = emit_expr(*x.m_id); }
    MLIR_ValueHandle pos = MLIR_INVALID_HANDLE;
        if (x.m_pos) { pos = emit_expr(*x.m_pos); }
    size_t n_values = x.n_values;
        MLIR_OpHandle *values = emit_expr_op_array(x.m_values, n_values);
    MLIR_ValueHandle overloaded = MLIR_INVALID_HANDLE;
        if (x.m_overloaded) { overloaded = emit_stmt(*x.m_overloaded); }
    bool is_formatted = x.m_is_formatted;
    string nml = str_lit("");
        if (x.m_nml) { nml = emit_symbol_ref(*x.m_nml); }
    MLIR_ValueHandle rec = MLIR_INVALID_HANDLE;
        if (x.m_rec) { rec = emit_expr(*x.m_rec); }
    MLIR_ValueHandle pad = MLIR_INVALID_HANDLE;
        if (x.m_pad) { pad = emit_expr(*x.m_pad); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFileReadOp(&ctx, op_loc, label, unit, fmt, iomsg, iostat, advance, size, id, pos, values, n_values, overloaded, is_formatted, nml, rec, pad);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_FileRewind(const ASR::FileRewind_t &x) {
    int64_t label = x.m_label;
    MLIR_ValueHandle unit = MLIR_INVALID_HANDLE;
        if (x.m_unit) { unit = emit_expr(*x.m_unit); }
    MLIR_ValueHandle iostat = MLIR_INVALID_HANDLE;
        if (x.m_iostat) { iostat = emit_expr(*x.m_iostat); }
    MLIR_ValueHandle err = MLIR_INVALID_HANDLE;
        if (x.m_err) { err = emit_expr(*x.m_err); }
    MLIR_ValueHandle iomsg = MLIR_INVALID_HANDLE;
        if (x.m_iomsg) { iomsg = emit_expr(*x.m_iomsg); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFileRewindOp(&ctx, op_loc, label, unit, iostat, err, iomsg);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Flush(const ASR::Flush_t &x) {
    int64_t label = x.m_label;
    MLIR_ValueHandle unit = emit_expr(*x.m_unit);
    MLIR_ValueHandle err = MLIR_INVALID_HANDLE;
        if (x.m_err) { err = emit_expr(*x.m_err); }
    MLIR_ValueHandle iomsg = MLIR_INVALID_HANDLE;
        if (x.m_iomsg) { iomsg = emit_expr(*x.m_iomsg); }
    MLIR_ValueHandle iostat = MLIR_INVALID_HANDLE;
        if (x.m_iostat) { iostat = emit_expr(*x.m_iostat); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateFlushOp(&ctx, op_loc, label, unit, err, iomsg, iostat);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ForAllSingle(const ASR::ForAllSingle_t &x) {
    MLIR_ValueHandle head = emit_do_loop_head(x.m_head);
    MLIR_ValueHandle assign_stmt = emit_stmt(*x.m_assign_stmt);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateForAllSingleOp(&ctx, op_loc, head, assign_stmt);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ForEach(const ASR::ForEach_t &x) {
    MLIR_ValueHandle var = emit_expr(*x.m_var);
    MLIR_ValueHandle container = emit_expr(*x.m_container);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_stmt_op_array(x.m_body, n_body);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateForEachOp(&ctx, op_loc, var, container, body, n_body);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_GoTo(const ASR::GoTo_t &x) {
    int64_t target_id = x.m_target_id;
    string name = asr_cstr(x.m_name);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateGoToOp(&ctx, op_loc, target_id, name);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_GoToTarget(const ASR::GoToTarget_t &x) {
    int64_t id = x.m_id;
    string name = asr_cstr(x.m_name);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateGoToTargetOp(&ctx, op_loc, id, name);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_GpuKernelLaunch(const ASR::GpuKernelLaunch_t &x) {
    string kernel = emit_symbol_ref(*x.m_kernel);
    MLIR_ValueHandle grid_size = emit_expr(*x.m_grid_size);
    MLIR_ValueHandle block_size = emit_expr(*x.m_block_size);
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_product_op_array(x.m_args, n_args);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateGpuKernelLaunchOp(&ctx, op_loc, kernel, grid_size, block_size, args, n_args);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_GpuSync(const ASR::GpuSync_t &x) {
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateGpuSyncOp(&ctx, op_loc);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_If(const ASR::If_t &x) {
    string name = asr_cstr(x.m_name);
    MLIR_ValueHandle test = emit_expr(*x.m_test);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_stmt_op_array(x.m_body, n_body);
    size_t n_orelse = x.n_orelse;
        MLIR_OpHandle *orelse = emit_stmt_op_array(x.m_orelse, n_orelse);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIfOp(&ctx, op_loc, name, test, body, n_body, orelse, n_orelse);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_IfArithmetic(const ASR::IfArithmetic_t &x) {
    MLIR_ValueHandle test = emit_expr(*x.m_test);
    int64_t lt_label = x.m_lt_label;
    int64_t eq_label = x.m_eq_label;
    int64_t gt_label = x.m_gt_label;
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIfArithmeticOp(&ctx, op_loc, test, lt_label, eq_label, gt_label);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ImplicitDeallocate(const ASR::ImplicitDeallocate_t &x) {
    size_t n_vars = x.n_vars;
        MLIR_OpHandle *vars = emit_expr_op_array(x.m_vars, n_vars);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateImplicitDeallocateOp(&ctx, op_loc, vars, n_vars);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_IntrinsicImpureSubroutine(const ASR::IntrinsicImpureSubroutine_t &x) {
    int64_t sub_intrinsic_id = x.m_sub_intrinsic_id;
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_expr_op_array(x.m_args, n_args);
    int64_t overload_id = x.m_overload_id;
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateIntrinsicImpureSubroutineOp(&ctx, op_loc, sub_intrinsic_id, args, n_args, overload_id);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ListAppend(const ASR::ListAppend_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle ele = emit_expr(*x.m_ele);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListAppendOp(&ctx, op_loc, a, ele);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ListClear(const ASR::ListClear_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListClearOp(&ctx, op_loc, a);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ListInsert(const ASR::ListInsert_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle pos = emit_expr(*x.m_pos);
    MLIR_ValueHandle ele = emit_expr(*x.m_ele);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListInsertOp(&ctx, op_loc, a, pos, ele);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ListRemove(const ASR::ListRemove_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle ele = emit_expr(*x.m_ele);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateListRemoveOp(&ctx, op_loc, a, ele);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Nullify(const ASR::Nullify_t &x) {
    size_t n_vars = x.n_vars;
        MLIR_OpHandle *vars = emit_expr_op_array(x.m_vars, n_vars);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateNullifyOp(&ctx, op_loc, vars, n_vars);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_OMPRegion(const ASR::OMPRegion_t &x) {
    int64_t region = (int64_t)x.m_region;
    size_t n_clauses = x.n_clauses;
        MLIR_OpHandle *clauses = emit_product_op_array(x.m_clauses, n_clauses);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_stmt_op_array(x.m_body, n_body);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateOMPRegionOp(&ctx, op_loc, region, clauses, n_clauses, body, n_body);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_ReAlloc(const ASR::ReAlloc_t &x) {
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_product_op_array(x.m_args, n_args);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateReAllocOp(&ctx, op_loc, args, n_args);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Return(const ASR::Return_t &x) {
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateReturnOp(&ctx, op_loc);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Select(const ASR::Select_t &x) {
    string name = asr_cstr(x.m_name);
    MLIR_ValueHandle test = emit_expr(*x.m_test);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_product_op_array(x.m_body, n_body);
    size_t n_default_ = x.n_default;
        MLIR_OpHandle *default_ = emit_stmt_op_array(x.m_default, n_default_);
    bool enable_fall_through = x.m_enable_fall_through;
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSelectOp(&ctx, op_loc, name, test, body, n_body, default_, n_default_, enable_fall_through);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_SelectRank(const ASR::SelectRank_t &x) {
    string name = asr_cstr(x.m_name);
    MLIR_ValueHandle selector = emit_expr(*x.m_selector);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_product_op_array(x.m_body, n_body);
    size_t n_default_ = x.n_default;
        MLIR_OpHandle *default_ = emit_stmt_op_array(x.m_default, n_default_);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSelectRankOp(&ctx, op_loc, name, selector, body, n_body, default_, n_default_);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_SelectType(const ASR::SelectType_t &x) {
    MLIR_ValueHandle selector = emit_expr(*x.m_selector);
    string assoc_name = asr_cstr(x.m_assoc_name);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_product_op_array(x.m_body, n_body);
    size_t n_default_ = x.n_default;
        MLIR_OpHandle *default_ = emit_stmt_op_array(x.m_default, n_default_);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSelectTypeOp(&ctx, op_loc, selector, assoc_name, body, n_body, default_, n_default_);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_SetClear(const ASR::SetClear_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSetClearOp(&ctx, op_loc, a);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_SetInsert(const ASR::SetInsert_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle ele = emit_expr(*x.m_ele);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSetInsertOp(&ctx, op_loc, a, ele);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_SetRemove(const ASR::SetRemove_t &x) {
    MLIR_ValueHandle a = emit_expr(*x.m_a);
    MLIR_ValueHandle ele = emit_expr(*x.m_ele);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSetRemoveOp(&ctx, op_loc, a, ele);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Stop(const ASR::Stop_t &x) {
    MLIR_ValueHandle code = MLIR_INVALID_HANDLE;
        if (x.m_code) { code = emit_expr(*x.m_code); }
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateStopOp(&ctx, op_loc, code);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_SubroutineCall(const ASR::SubroutineCall_t &x) {
    string name = emit_symbol_ref(*x.m_name);
    string original_name = str_lit("");
        if (x.m_original_name) { original_name = emit_symbol_ref(*x.m_original_name); }
    size_t n_args = x.n_args;
        MLIR_OpHandle *args = emit_product_op_array(x.m_args, n_args);
    MLIR_ValueHandle dt = MLIR_INVALID_HANDLE;
        if (x.m_dt) { dt = emit_expr(*x.m_dt); }
    bool strict_bounds_checking = x.m_strict_bounds_checking;
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSubroutineCallOp(&ctx, op_loc, name, original_name, args, n_args, dt, strict_bounds_checking);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_SyncAll(const ASR::SyncAll_t &x) {
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSyncAllOp(&ctx, op_loc);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_SyncMemory(const ASR::SyncMemory_t &x) {
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateSyncMemoryOp(&ctx, op_loc);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_Where(const ASR::Where_t &x) {
    MLIR_ValueHandle test = emit_expr(*x.m_test);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_stmt_op_array(x.m_body, n_body);
    size_t n_orelse = x.n_orelse;
        MLIR_OpHandle *orelse = emit_stmt_op_array(x.m_orelse, n_orelse);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateWhereOp(&ctx, op_loc, test, body, n_body, orelse, n_orelse);
    append_current_stmt(last_value);
}

void ASRToAsrDialectVisitor::visit_WhileLoop(const ASR::WhileLoop_t &x) {
    string name = asr_cstr(x.m_name);
    MLIR_ValueHandle test = emit_expr(*x.m_test);
    size_t n_body = x.n_body;
        MLIR_OpHandle *body = emit_stmt_op_array(x.m_body, n_body);
    size_t n_orelse = x.n_orelse;
        MLIR_OpHandle *orelse = emit_stmt_op_array(x.m_orelse, n_orelse);
    MLIR_LocationHandle op_loc = loc(x.base.base.loc);
    last_value = ASR_CreateWhileLoopOp(&ctx, op_loc, name, test, body, n_body, orelse, n_orelse);
    append_current_stmt(last_value);
}
// <<< END GENERATED VISITOR IMPLEMENTATIONS <<<

void ASRToAsrDialectVisitor::visit_Function(const ASR::Function_t &x) {
    ASR::FunctionType_t *fn_type =
        ASR::down_cast<ASR::FunctionType_t>(x.m_function_signature);
    if (fn_type->m_deftype == ASR::deftypeType::Interface) {
        return;
    }

    std::vector<MLIR_OpHandle> saved_symtab = scope_symtab;
    std::vector<MLIR_OpHandle> saved_metadata = scope_metadata;
    std::vector<MLIR_OpHandle> saved_body = scope_body;
    bool saved_terminated = block_terminated;
    ScopeRegion saved_region = scope_region;

    scope_symtab.clear();
    scope_metadata.clear();
    scope_body.clear();
    block_terminated = false;

    MLIR_TypeHandle fn_sig = convert_type(*x.m_function_signature);
    string deps = emit_identifier_seq(x.m_dependencies, x.n_dependencies);
    MLIR_OpHandle fn_op = ASR_CreateFunctionOp(&ctx, default_loc(),
        MLIR_INVALID_HANDLE, asr_cstr(x.m_name), fn_sig, deps,
        nullptr, 0, nullptr, 0, MLIR_INVALID_HANDLE,
        (int64_t)x.m_access, x.m_deterministic, x.m_side_effect_free,
        x.m_module_file ? asr_cstr(x.m_module_file) : str_lit(""),
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE);

    scope_region = ScopeRegion::Symtab;
    for (auto &item : x.m_symtab->get_scope()) {
        if (ASR::is_a<ASR::Variable_t>(*item.second)) {
            ASR::Variable_t *v = ASR::down_cast<ASR::Variable_t>(item.second);
            if (v->m_intent == ASR::intentType::Local ||
                    v->m_intent == ASR::intentType::ReturnVar) {
                visit_Variable(*v);
            }
        }
    }

    scope_region = ScopeRegion::Metadata;

    scope_region = ScopeRegion::Body;
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

    attach_scope_regions(fn_op);

    scope_symtab = saved_symtab;
    scope_metadata = saved_metadata;
    scope_body = saved_body;
    block_terminated = saved_terminated;
    scope_region = saved_region;

    last_value = fn_op;
    append_scope_op(fn_op);
}

void ASRToAsrDialectVisitor::visit_Program(const ASR::Program_t &x) {
    scope_symtab.clear();
    scope_metadata.clear();
    scope_body.clear();
    block_terminated = false;

    program_op = ASR_CreateProgramOp(&ctx, default_loc(),
        MLIR_INVALID_HANDLE, asr_cstr(x.m_name), str_lit(""),
        nullptr, 0, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE);

    scope_region = ScopeRegion::Symtab;
    for (auto &item : x.m_symtab->get_scope()) {
        if (ASR::is_a<ASR::Variable_t>(*item.second)) {
            visit_Variable(*ASR::down_cast<ASR::Variable_t>(item.second));
        } else if (ASR::is_a<ASR::Function_t>(*item.second)) {
            visit_Function(*ASR::down_cast<ASR::Function_t>(item.second));
        }
    }

    scope_region = ScopeRegion::Metadata;
    // Bulky statement metadata (IO, assignment info, labels) is emitted here later.

    scope_region = ScopeRegion::Body;
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

    attach_scope_regions(program_op);
    scope_region = ScopeRegion::Module;
    last_value = MLIR_INVALID_HANDLE;
}

void ASRToAsrDialectVisitor::visit_TranslationUnit(const ASR::TranslationUnit_t &x) {
    emit_module_skeleton();
    ASR_DialectModuleStorageInit(&ctx);
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
    append_module(last_value);
}

} // namespace LCompilers
