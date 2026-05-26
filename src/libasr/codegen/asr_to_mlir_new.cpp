#include <libasr/codegen/asr_to_mlir_new.h>
#include <libasr/diagnostics.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include <mlir_api.h>
#include <base/arena.h>
#include <base/string.h>
#include <platform/platform.h>
}

namespace LCompilers {

std::once_flag mlir_corec_platform_init;

void ensure_mlir_corec_platform_initialized() {
    std::call_once(mlir_corec_platform_init,
                   []() { platform_init(0, nullptr); });
}

uint64_t node_hash(const ASR::asr_t *node) { return (uint64_t)node; }

class CodeGenError {
public:
    diag::Diagnostic d;
    explicit CodeGenError(const std::string &msg)
        : d{diag::Diagnostic(msg, diag::Level::Error, diag::Stage::CodeGen)} {}
    CodeGenError(const std::string &msg, const Location &loc)
        : d{diag::Diagnostic(msg, diag::Level::Error, diag::Stage::CodeGen,
              {diag::Label("", {loc})})} {}
};

static std::string copy_mlir_string(string s) {
    if (!s.str || s.size == 0) {
        return std::string();
    }
    return std::string(s.str, s.str + s.size);
}

static string arena_ssa_name(Arena *arena, int id) {
    char *buf = (char *)arena_alloc(arena, 48);
    int n = snprintf(buf, 48, "v%d", id);
    if (n < 0) {
        std::abort();
    }
    return str_from_cstr_len_view(buf, (uint64_t)n);
}

static const int32_t LLVM_GEP_DYN = (int32_t)0x80000000;

// LLVM ICmpPredicate (signed): eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5
static int64_t icmp_predicate_for(ASR::cmpopType op) {
    switch (op) {
        case ASR::cmpopType::Eq:
            return 0;
        case ASR::cmpopType::NotEq:
            return 1;
        case ASR::cmpopType::Lt:
            return 2;
        case ASR::cmpopType::LtE:
            return 3;
        case ASR::cmpopType::Gt:
            return 4;
        case ASR::cmpopType::GtE:
            return 5;
        default:
            return -1;
    }
}

struct SymSlot {
    MLIR_ValueHandle ptr = MLIR_INVALID_HANDLE;
    bool is_array = false;
    int64_t array_len = 0;
    MLIR_TypeHandle llvm_array_ty = MLIR_INVALID_HANDLE;
};

struct LlvmEmitter {
    Arena *arena = nullptr;
    MLIR_Context ctx{};
    MLIR_LocationHandle loc = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle i32_ty = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle i64_ty = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle i1_ty = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle ptr_ty = MLIR_INVALID_HANDLE;
    MLIR_TypeHandle void_ty = MLIR_INVALID_HANDLE;
    MLIR_BlockHandle module_block = MLIR_INVALID_HANDLE;
    MLIR_BlockHandle cur_block = MLIR_INVALID_HANDLE;
    MLIR_RegionHandle fn_body_r = MLIR_INVALID_HANDLE;
    int ssa_counter = 0;
    std::map<uint64_t, SymSlot> symtab;
    MLIR_ValueHandle last_value = MLIR_INVALID_HANDLE;
    MLIR_OpHandle module_op = MLIR_INVALID_HANDLE;
    MLIR_ValueHandle entry_const_one = MLIR_INVALID_HANDLE;
    bool block_terminated = false;
    bool printf_declared = false;
    bool fmt_global_added = false;

    static void check_mlir(bool ok, const char *stage) {
        if (!ok) {
            throw CodeGenError(std::string("mlir-new: failed at ") + stage);
        }
    }

    MLIR_OpHandle emit_simple_op(MLIR_OpType ty, string opname,
            MLIR_AttributeHandle *attrs, size_t n_attrs,
            MLIR_TypeHandle *result_types, size_t n_result_types,
            MLIR_ValueHandle *results, size_t n_results,
            MLIR_ValueHandle *operands, size_t n_operands,
            MLIR_RegionHandle *regions, size_t n_regions) {
        return MLIR_CreateOp(&ctx, ty, opname, attrs, n_attrs, result_types,
            n_result_types, results, n_results, operands, n_operands, regions,
            n_regions, loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    }

    void append_current(MLIR_OpHandle op) {
        check_mlir(cur_block != MLIR_INVALID_HANDLE, "missing current block");
        check_mlir(!block_terminated, "appending to terminated block");
        MLIR_AppendBlockOp(&ctx, cur_block, op);
    }

    void append_module(MLIR_OpHandle op) {
        check_mlir(module_block != MLIR_INVALID_HANDLE, "missing module block");
        MLIR_AppendBlockOp(&ctx, module_block, op);
    }

    MLIR_BlockHandle new_cfg_block() {
        check_mlir(fn_body_r != MLIR_INVALID_HANDLE, "missing function region");
        MLIR_BlockHandle b = MLIR_CreateBlock(&ctx);
        MLIR_AppendRegionBlock(&ctx, fn_body_r, b);
        return b;
    }

    void emit_branch(MLIR_BlockHandle target) {
        if (block_terminated) {
            return;
        }
        MLIR_BlockHandle succs[1] = {target};
        MLIR_ValueHandle *sops[1] = {nullptr};
        size_t snums[1] = {0};
        MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(
            &ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.br"),
            nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0,
            succs, 1, sops, snums,
            loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        append_current(op);
        block_terminated = true;
    }

    void emit_cond_branch(MLIR_ValueHandle cond, MLIR_BlockHandle true_b,
            MLIR_BlockHandle false_b) {
        if (block_terminated) {
            return;
        }
        MLIR_ValueHandle ops[1] = {cond};
        MLIR_BlockHandle succs[2] = {true_b, false_b};
        MLIR_ValueHandle *sops[2] = {nullptr, nullptr};
        size_t snums[2] = {0, 0};
        MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(
            &ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.cond_br"),
            nullptr, 0, nullptr, 0, nullptr, 0, ops, 1, nullptr, 0,
            succs, 2, sops, snums,
            loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        append_current(op);
        block_terminated = true;
    }

    void emit_module_skeleton() {
        arena = arena_create(131072);
        if (!arena) {
            throw CodeGenError("mlir-new: arena_create failed");
        }
        MLIR_SetArenaAllocator(&ctx, arena);
        loc = MLIR_CreateLocationUnknown(&ctx, str_lit("lfortran"));
        i32_ty = MLIR_CreateTypeInteger(&ctx, 32, true);
        i64_ty = MLIR_CreateTypeInteger(&ctx, 64, false);
        i1_ty = MLIR_CreateTypeInteger(&ctx, 1, false);
        ptr_ty = MLIR_CreateTypeLLVMPointer(&ctx);
        void_ty = MLIR_CreateTypeLLVMVoid(&ctx);

        MLIR_RegionHandle mr = MLIR_CreateRegion(&ctx);
        module_block = MLIR_CreateBlock(&ctx);
        MLIR_AppendRegionBlock(&ctx, mr, module_block);
        MLIR_RegionHandle mregs[1] = {mr};
        module_op = MLIR_CreateOp(&ctx, OP_TYPE_MODULE, str_lit("module"),
            nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, mregs, 1,
            loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        check_mlir(module_op != MLIR_INVALID_HANDLE, "builtin.module");
    }

    MLIR_ValueHandle emit_const_i32(int64_t value) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(
            &ctx, MLIR_INVALID_HANDLE, 0, i32_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i32_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_AttributeHandle val =
            MLIR_CreateAttributeInteger(&ctx, str_lit("value"), value, i32_ty);
        MLIR_AttributeHandle as[1] = {val};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_MLIR_CONSTANT,
            str_lit("llvm.mlir.constant"), as, 1, rts, 1, rs, 1, nullptr, 0,
            nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_const_i64(int64_t value) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(
            &ctx, MLIR_INVALID_HANDLE, 0, i64_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i64_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_AttributeHandle val =
            MLIR_CreateAttributeInteger(&ctx, str_lit("value"), value, i64_ty);
        MLIR_AttributeHandle as[1] = {val};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_MLIR_CONSTANT,
            str_lit("llvm.mlir.constant"), as, 1, rts, 1, rs, 1, nullptr, 0,
            nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_alloca(MLIR_TypeHandle elem_ty) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, ptr_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {ptr_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[1] = {entry_const_one};
        MLIR_AttributeHandle elem_attr =
            MLIR_CreateAttributeType(&ctx, str_lit("elem_type"), elem_ty);
        MLIR_AttributeHandle as[1] = {elem_attr};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_ALLOCA,
            str_lit("llvm.alloca"), as, 1, rts, 1, rs, 1, ops, 1, nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_gep(MLIR_ValueHandle base,
            MLIR_TypeHandle source_elem_ty, const int32_t *raw_idx,
            size_t n_raw, MLIR_ValueHandle *dyn_idx, size_t n_dyn) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, ptr_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {ptr_ty};
        MLIR_ValueHandle rs[1] = {res};
        std::vector<MLIR_ValueHandle> ops;
        ops.push_back(base);
        for (size_t i = 0; i < n_dyn; i++) {
            ops.push_back(dyn_idx[i]);
        }
        MLIR_AttributeHandle raw_attr = MLIR_CreateAttributeDenseI32Array(
            &ctx, str_lit("rawConstantIndices"), raw_idx, n_raw);
        MLIR_AttributeHandle elem_attr =
            MLIR_CreateAttributeType(&ctx, str_lit("elem_type"), source_elem_ty);
        MLIR_AttributeHandle as[2] = {raw_attr, elem_attr};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_GEP,
            str_lit("llvm.getelementptr"), as, 2, rts, 1, rs, 1,
            ops.data(), ops.size(), nullptr, 0);
        append_current(op);
        return res;
    }

    void emit_store_i32(MLIR_ValueHandle val, MLIR_ValueHandle ptr) {
        MLIR_ValueHandle ops[2] = {val, ptr};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_STORE,
            str_lit("llvm.store"), nullptr, 0, nullptr, 0, nullptr, 0, ops, 2,
            nullptr, 0);
        append_current(op);
    }

    MLIR_ValueHandle emit_load_i32(MLIR_ValueHandle ptr) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i32_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i32_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[1] = {ptr};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_LOAD,
            str_lit("llvm.load"), nullptr, 0, rts, 1, rs, 1, ops, 1, nullptr,
            0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_binop_i32(MLIR_OpType ty, string opname,
            MLIR_ValueHandle a, MLIR_ValueHandle b) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i32_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i32_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[2] = {a, b};
        MLIR_OpHandle op = emit_simple_op(ty, opname, nullptr, 0, rts, 1, rs,
            1, ops, 2, nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_binop_i1(MLIR_OpType ty, string opname,
            MLIR_ValueHandle a, MLIR_ValueHandle b) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i1_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i1_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[2] = {a, b};
        MLIR_OpHandle op = emit_simple_op(ty, opname, nullptr, 0, rts, 1, rs,
            1, ops, 2, nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_const_i1(bool value) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(
            &ctx, MLIR_INVALID_HANDLE, 0, i1_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i1_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_AttributeHandle val = MLIR_CreateAttributeInteger(
            &ctx, str_lit("value"), value ? 1 : 0, i1_ty);
        MLIR_AttributeHandle as[1] = {val};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_MLIR_CONSTANT,
            str_lit("llvm.mlir.constant"), as, 1, rts, 1, rs, 1, nullptr, 0,
            nullptr, 0);
        append_current(op);
        return res;
    }

    MLIR_ValueHandle emit_icmp_i32(int64_t predicate, MLIR_ValueHandle a,
            MLIR_ValueHandle b) {
        MLIR_ValueHandle res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i1_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {i1_ty};
        MLIR_ValueHandle rs[1] = {res};
        MLIR_ValueHandle ops[2] = {a, b};
        MLIR_AttributeHandle pred = MLIR_CreateAttributeInteger(&ctx,
            str_lit("predicate"), predicate, i64_ty);
        MLIR_AttributeHandle as[1] = {pred};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_ICMP,
            str_lit("llvm.icmp"), as, 1, rts, 1, rs, 1, ops, 2, nullptr, 0);
        append_current(op);
        return res;
    }

    void emit_return_i32(MLIR_ValueHandle ret) {
        MLIR_ValueHandle ops[1] = {ret};
        MLIR_OpHandle op = emit_simple_op(OP_TYPE_LLVM_RETURN,
            str_lit("llvm.return"), nullptr, 0, nullptr, 0, nullptr, 0, ops,
            1, nullptr, 0);
        append_current(op);
        block_terminated = true;
    }

    MLIR_OpHandle emit_llvm_func(string sym, MLIR_TypeHandle fn_ty,
            MLIR_RegionHandle body_r) {
        MLIR_AttributeHandle sym_name =
            MLIR_CreateAttributeString(&ctx, str_lit("sym_name"), sym);
        MLIR_AttributeHandle fn_ty_attr =
            MLIR_CreateAttributeType(&ctx, str_lit("function_type"), fn_ty);
        MLIR_AttributeHandle attrs[2] = {sym_name, fn_ty_attr};
        MLIR_RegionHandle regs[1] = {body_r};
        // Match tinyc: external/vararg decls use llvm.func via unregistered
        // op name with an (possibly empty) body region attached.
        MLIR_OpHandle fn = MLIR_CreateOp(&ctx, OP_TYPE_UNREGISTERED,
            str_lit("llvm.func"), attrs, 2,
            nullptr, 0, nullptr, 0, nullptr, 0,
            regs, 1, loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        check_mlir(fn != MLIR_INVALID_HANDLE, "llvm.func");
        return fn;
    }

    void emit_module_runtime_decls() {
        if (!printf_declared) {
            MLIR_TypeHandle fn_ty = MLIR_CreateTypeLLVMFunction(&ctx, i32_ty,
                &ptr_ty, 1, true);
            MLIR_RegionHandle empty_r = MLIR_CreateRegion(&ctx);
            append_module(emit_llvm_func(str_lit("printf"), fn_ty, empty_r));
            printf_declared = true;
        }
        if (!fmt_global_added) {
            MLIR_OpHandle fmt_global = MLIR_CreateLLVMGlobalString(&ctx,
                str_lit(".fmt.int"), str_lit("%d\n"), loc);
            append_module(fmt_global);
            fmt_global_added = true;
        }
    }

    void declare_printf() {
        emit_module_runtime_decls();
    }

    void emit_printf_i32(MLIR_ValueHandle val) {
        emit_module_runtime_decls();
        MLIR_ValueHandle fmt_ptr = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, ptr_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle rts[1] = {ptr_ty};
        MLIR_ValueHandle rs[1] = {fmt_ptr};
        MLIR_AttributeHandle global_name = MLIR_CreateAttributeSymbolRef(&ctx,
            str_lit("global_name"), str_lit(".fmt.int"));
        MLIR_AttributeHandle as[1] = {global_name};
        MLIR_OpHandle addr = emit_simple_op(OP_TYPE_LLVM_MLIR_ADDRESSOF,
            str_lit("llvm.mlir.addressof"), as, 1, rts, 1, rs, 1, nullptr, 0,
            nullptr, 0);
        append_current(addr);

        MLIR_TypeHandle call_ty = MLIR_CreateTypeLLVMFunction(&ctx, i32_ty,
            &ptr_ty, 1, true);
        MLIR_AttributeHandle callee =
            MLIR_CreateAttributeSymbolRef(&ctx, str_lit("callee"), str_lit("printf"));
        MLIR_AttributeHandle var_ty = MLIR_CreateAttributeType(&ctx,
            str_lit("var_callee_type"), call_ty);
        MLIR_AttributeHandle call_attrs[2] = {callee, var_ty};
        MLIR_ValueHandle ops[2] = {fmt_ptr, val};
        // Variadic llvm.call must expose the callee return type as an op
        // result for upstream translateModuleToLLVMIR (tinyc pattern).
        MLIR_ValueHandle call_res = MLIR_CreateValueOpResult(&ctx,
            MLIR_INVALID_HANDLE, 0, i32_ty,
            arena_ssa_name(arena, ssa_counter++), loc);
        MLIR_TypeHandle call_rts[1] = {i32_ty};
        MLIR_ValueHandle call_rs[1] = {call_res};
        MLIR_OpHandle call = emit_simple_op(OP_TYPE_UNREGISTERED,
            str_lit("llvm.call"), call_attrs, 2, call_rts, 1, call_rs, 1, ops,
            2, nullptr, 0);
        append_current(call);
        (void)call_res;
    }

    bool is_supported_integer_type(ASR::ttype_t *type) {
        type = ASRUtils::type_get_past_allocatable(
            ASRUtils::type_get_past_pointer(type));
        return ASRUtils::is_integer(*type);
    }

    bool is_fixed_integer_array(ASR::ttype_t *type, int64_t &len) {
        type = ASRUtils::type_get_past_allocatable(
            ASRUtils::type_get_past_pointer(type));
        if (!ASR::is_a<ASR::Array_t>(*type)) {
            return false;
        }
        if (!ASRUtils::is_fixed_size_array(type)) {
            return false;
        }
        ASR::ttype_t *elem = ASRUtils::type_get_past_array(type);
        if (!ASRUtils::is_integer(*elem)) {
            return false;
        }
        len = ASRUtils::get_fixed_size_of_array(type);
        return len > 0;
    }

    SymSlot &require_slot(uint64_t h, const Location &loc) {
        auto it = symtab.find(h);
        if (it == symtab.end()) {
            throw CodeGenError("symbol not found", loc);
        }
        return it->second;
    }

    SymSlot &slot_for_var(ASR::Variable_t *v) {
        return require_slot(node_hash(reinterpret_cast<const ASR::asr_t *>(v)),
            v->base.base.loc);
    }

    MLIR_ValueHandle fortran_to_zero_based_index(ASR::expr_t *idx_expr,
            ASR::dimension_t *dim) {
        visit_expr_value(*idx_expr);
        MLIR_ValueHandle fortran_idx = last_value;
        int64_t lbound = 1;
        if (dim && dim->m_start) {
            if (!ASRUtils::extract_value(ASRUtils::expr_value(dim->m_start),
                    lbound)) {
                throw CodeGenError(
                    "mlir-new: array lower bound must be constant for now",
                    idx_expr->base.loc);
            }
        }
        if (lbound == 0) {
            return fortran_idx;
        }
        MLIR_ValueHandle lb = emit_const_i32(lbound);
        return emit_binop_i32(OP_TYPE_LLVM_SUB, str_lit("llvm.sub"),
            fortran_idx, lb);
    }

    MLIR_ValueHandle gep_array_element(SymSlot &slot,
            MLIR_ValueHandle zero_based_idx) {
        int32_t path[2] = {0, LLVM_GEP_DYN};
        return emit_gep(slot.ptr, slot.llvm_array_ty, path, 2,
            &zero_based_idx, 1);
    }

    MLIR_ValueHandle ptr_for_array_item(const ASR::ArrayItem_t &x) {
        if (!ASR::is_a<ASR::Var_t>(*x.m_v)) {
            throw CodeGenError(
                "mlir-new: array base must be a variable for now",
                x.base.base.loc);
        }
        ASR::Variable_t *v = ASRUtils::EXPR2VAR(x.m_v);
        SymSlot &slot = slot_for_var(v);
        if (!slot.is_array) {
            throw CodeGenError("mlir-new: indexing a non-array variable",
                x.base.base.loc);
        }
        if (x.n_args != 1) {
            throw CodeGenError(
                "mlir-new: only rank-1 array indexing is supported for now",
                x.base.base.loc);
        }
        ASR::dimension_t *dims = nullptr;
        size_t n_dims = ASRUtils::extract_dimensions_from_ttype(v->m_type, dims);
        ASR::dimension_t *dim = (n_dims > 0) ? &dims[0] : nullptr;
        MLIR_ValueHandle idx0 = fortran_to_zero_based_index(
            x.m_args[0].m_right, dim);
        return gep_array_element(slot, idx0);
    }

    void emit_variable_storage(const ASR::Variable_t &x) {
        uint64_t h = node_hash(reinterpret_cast<const ASR::asr_t *>(&x));
        int64_t array_len = 0;
        if (is_fixed_integer_array(x.m_type, array_len)) {
            MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                &ctx, i32_ty, (uint64_t)array_len);
            SymSlot slot;
            slot.is_array = true;
            slot.array_len = array_len;
            slot.llvm_array_ty = arr_ty;
            slot.ptr = emit_alloca(arr_ty);
            symtab[h] = slot;
            return;
        }
        if (!is_supported_integer_type(x.m_type)) {
            throw CodeGenError(
                "mlir-new: only integer scalars and fixed-size integer arrays "
                "are supported for now",
                x.base.base.loc);
        }
        SymSlot slot;
        slot.ptr = emit_alloca(i32_ty);
        symtab[h] = slot;
    }

    void init_array_from_constant(const ASR::Variable_t &x, SymSlot &slot,
            const ASR::ArrayConstant_t &ac) {
        if (!slot.is_array) {
            throw CodeGenError("mlir-new: internal array init mismatch",
                x.base.base.loc);
        }
        int64_t n_elem = ac.m_n_data;
        if (n_elem > slot.array_len) {
            n_elem = slot.array_len;
        }
        ASR::ttype_t *elem_type = ASRUtils::type_get_past_array(ac.m_type);
        if (!ASRUtils::is_integer(*elem_type)) {
            throw CodeGenError(
                "mlir-new: only integer array constants are supported",
                x.base.base.loc);
        }
        int64_t kind = ASRUtils::extract_kind_from_ttype_t(elem_type);
        for (int64_t i = 0; i < n_elem; i++) {
            int64_t value = 0;
            if (kind == 4) {
                value = ((int32_t *)ac.m_data)[i];
            } else if (kind == 8) {
                value = ((int64_t *)ac.m_data)[i];
            } else {
                throw CodeGenError(
                    "mlir-new: unsupported integer kind in array constant",
                    x.base.base.loc);
            }
            MLIR_ValueHandle idx = emit_const_i32(i);
            MLIR_ValueHandle elem_ptr = gep_array_element(slot, idx);
            MLIR_ValueHandle c = emit_const_i32(value);
            emit_store_i32(c, elem_ptr);
        }
    }

    void emit_variable_static_init(const ASR::Variable_t &x) {
        if (!x.m_symbolic_value) {
            return;
        }
        SymSlot &slot = slot_for_var(
            const_cast<ASR::Variable_t *>(&x));
        if (ASR::is_a<ASR::ArrayConstant_t>(*x.m_symbolic_value)) {
            init_array_from_constant(x, slot,
                *ASR::down_cast<ASR::ArrayConstant_t>(x.m_symbolic_value));
            return;
        }
        visit_expr_value(*x.m_symbolic_value);
        emit_store_i32(last_value, slot.ptr);
    }

    void visit_IntegerBinOp(const ASR::IntegerBinOp_t &x) {
        visit_expr_value(*x.m_left);
        MLIR_ValueHandle lhs = last_value;
        visit_expr_value(*x.m_right);
        MLIR_ValueHandle rhs = last_value;
        switch (x.m_op) {
            case ASR::binopType::Add:
                last_value = emit_binop_i32(OP_TYPE_LLVM_ADD,
                    str_lit("llvm.add"), lhs, rhs);
                return;
            case ASR::binopType::Sub:
                last_value = emit_binop_i32(OP_TYPE_LLVM_SUB,
                    str_lit("llvm.sub"), lhs, rhs);
                return;
            case ASR::binopType::Mul:
                last_value = emit_binop_i32(OP_TYPE_LLVM_MUL,
                    str_lit("llvm.mul"), lhs, rhs);
                return;
            case ASR::binopType::Div:
                last_value = emit_binop_i32(OP_TYPE_LLVM_SDIV,
                    str_lit("llvm.sdiv"), lhs, rhs);
                return;
            default:
                throw CodeGenError(
                    "mlir-new: integer binary operator not supported yet",
                    x.base.base.loc);
        }
    }

    void visit_expr_value(ASR::expr_t &e) {
        if (ASR::is_a<ASR::IntegerConstant_t>(e)) {
            const ASR::IntegerConstant_t *ic =
                ASR::down_cast<ASR::IntegerConstant_t>(&e);
            last_value = emit_const_i32(ic->m_n);
            return;
        }
        if (ASR::is_a<ASR::Var_t>(e)) {
            ASR::Variable_t *v = ASRUtils::EXPR2VAR(&e);
            SymSlot &slot = slot_for_var(v);
            if (slot.is_array) {
                throw CodeGenError(
                    "mlir-new: use array element in expression context",
                    e.base.loc);
            }
            last_value = emit_load_i32(slot.ptr);
            return;
        }
        if (ASR::is_a<ASR::ArrayItem_t>(e)) {
            MLIR_ValueHandle elem_ptr = ptr_for_array_item(
                *ASR::down_cast<ASR::ArrayItem_t>(&e));
            last_value = emit_load_i32(elem_ptr);
            return;
        }
        if (ASR::is_a<ASR::IntegerBinOp_t>(e)) {
            visit_IntegerBinOp(*ASR::down_cast<ASR::IntegerBinOp_t>(&e));
            return;
        }
        throw CodeGenError(
            "mlir-new: expression kind not supported yet", e.base.loc);
    }

    MLIR_ValueHandle visit_expr_bool(ASR::expr_t &e) {
        if (ASR::is_a<ASR::LogicalConstant_t>(e)) {
            const ASR::LogicalConstant_t *lc =
                ASR::down_cast<ASR::LogicalConstant_t>(&e);
            return emit_const_i1(lc->m_value);
        }
        if (ASR::is_a<ASR::IntegerCompare_t>(e)) {
            const ASR::IntegerCompare_t *ic =
                ASR::down_cast<ASR::IntegerCompare_t>(&e);
            int64_t pred = icmp_predicate_for(ic->m_op);
            if (pred < 0) {
                throw CodeGenError(
                    "mlir-new: integer compare operator not supported",
                    e.base.loc);
            }
            visit_expr_value(*ic->m_left);
            MLIR_ValueHandle lhs = last_value;
            visit_expr_value(*ic->m_right);
            MLIR_ValueHandle rhs = last_value;
            return emit_icmp_i32(pred, lhs, rhs);
        }
        if (ASR::is_a<ASR::LogicalBinOp_t>(e)) {
            const ASR::LogicalBinOp_t *lb =
                ASR::down_cast<ASR::LogicalBinOp_t>(&e);
            MLIR_ValueHandle lhs = visit_expr_bool(*lb->m_left);
            MLIR_ValueHandle rhs = visit_expr_bool(*lb->m_right);
            switch (lb->m_op) {
                case ASR::logicalbinopType::And:
                    return emit_binop_i1(OP_TYPE_LLVM_AND,
                        str_lit("llvm.and"), lhs, rhs);
                case ASR::logicalbinopType::Or:
                    return emit_binop_i1(OP_TYPE_LLVM_OR,
                        str_lit("llvm.or"), lhs, rhs);
                default:
                    throw CodeGenError(
                        "mlir-new: logical binary operator not supported",
                        e.base.loc);
            }
        }
        throw CodeGenError(
            "mlir-new: boolean expression kind not supported yet", e.base.loc);
    }

    void store_to_target(ASR::expr_t *target) {
        if (ASR::is_a<ASR::Var_t>(*target)) {
            ASR::Variable_t *v = ASRUtils::EXPR2VAR(target);
            SymSlot &slot = slot_for_var(v);
            if (slot.is_array) {
                throw CodeGenError(
                    "mlir-new: assign whole array not supported yet",
                    target->base.loc);
            }
            emit_store_i32(last_value, slot.ptr);
            return;
        }
        if (ASR::is_a<ASR::ArrayItem_t>(*target)) {
            MLIR_ValueHandle elem_ptr = ptr_for_array_item(
                *ASR::down_cast<ASR::ArrayItem_t>(target));
            emit_store_i32(last_value, elem_ptr);
            return;
        }
        throw CodeGenError("mlir-new: assignment target not supported",
            target->base.loc);
    }

    void visit_Assignment(const ASR::Assignment_t &x) {
        visit_expr_value(*x.m_value);
        store_to_target(x.m_target);
    }

    void handle_print_formatter(ASR::expr_t *x) {
        if (ASR::is_a<ASR::StringFormat_t>(*x)) {
            const ASR::StringFormat_t *sf =
                ASR::down_cast<ASR::StringFormat_t>(x);
            for (size_t i = 0; i < sf->n_args; i++) {
                visit_expr_value(*sf->m_args[i]);
                if (!ASRUtils::is_integer(
                        *ASRUtils::expr_type(sf->m_args[i]))) {
                    throw CodeGenError(
                        "mlir-new: print only supports integers in this "
                        "minimal pipeline",
                        sf->m_args[i]->base.loc);
                }
                emit_printf_i32(last_value);
            }
            return;
        }
        ASR::ttype_t *t = ASRUtils::expr_type(x);
        if (ASRUtils::is_integer(*t)) {
            visit_expr_value(*x);
            emit_printf_i32(last_value);
            return;
        }
        throw CodeGenError("mlir-new: print formatter not supported",
            x->base.loc);
    }

    void visit_Print(const ASR::Print_t &x) {
        if (!x.m_text) {
            throw CodeGenError("mlir-new: print with no format expression",
                x.base.base.loc);
        }
        handle_print_formatter(x.m_text);
    }

    void visit_FileWrite(const ASR::FileWrite_t &x) {
        bool default_unit = false;
        if (x.m_unit) {
            int uv = -1;
            if (ASRUtils::extract_value(x.m_unit, uv) && uv == 6) {
                default_unit = true;
            }
        } else {
            default_unit = true;
        }
        if (!default_unit) {
            throw CodeGenError(
                "mlir-new: only write to default unit (*,*) is supported",
                x.base.base.loc);
        }
        if (x.n_values != 1) {
            throw CodeGenError("mlir-new: expected a single format/value in "
                               "write(*,*) for minimal pipeline",
                x.base.base.loc);
        }
        handle_print_formatter(x.m_values[0]);
    }

    void visit_DoLoop(const ASR::DoLoop_t &x) {
        if (x.n_orelse > 0) {
            throw CodeGenError("mlir-new: do-loop else not supported",
                x.base.base.loc);
        }
        if (!x.m_head.m_v || !x.m_head.m_start || !x.m_head.m_end) {
            throw CodeGenError("mlir-new: incomplete do-loop header",
                x.base.base.loc);
        }
        if (!ASR::is_a<ASR::Var_t>(*x.m_head.m_v)) {
            throw CodeGenError("mlir-new: do-loop index must be a variable",
                x.base.base.loc);
        }
        ASR::Variable_t *loop_var = ASRUtils::EXPR2VAR(x.m_head.m_v);
        SymSlot &slot = slot_for_var(loop_var);
        if (slot.is_array) {
            throw CodeGenError("mlir-new: do-loop index must be scalar",
                x.base.base.loc);
        }

        visit_expr_value(*x.m_head.m_start);
        emit_store_i32(last_value, slot.ptr);

        MLIR_BlockHandle header_b = new_cfg_block();
        MLIR_BlockHandle body_b = new_cfg_block();
        MLIR_BlockHandle step_b = new_cfg_block();
        MLIR_BlockHandle exit_b = new_cfg_block();
        emit_branch(header_b);

        cur_block = header_b;
        block_terminated = false;
        MLIR_ValueHandle iv = emit_load_i32(slot.ptr);
        visit_expr_value(*x.m_head.m_end);
        MLIR_ValueHandle endv = last_value;
        MLIR_ValueHandle cond = emit_icmp_i32(
            icmp_predicate_for(ASR::cmpopType::LtE), iv, endv);
        emit_cond_branch(cond, body_b, exit_b);

        cur_block = body_b;
        block_terminated = false;
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
            if (block_terminated) {
                break;
            }
        }
        if (!block_terminated) {
            emit_branch(step_b);
        }

        cur_block = step_b;
        block_terminated = false;
        iv = emit_load_i32(slot.ptr);
        MLIR_ValueHandle stepv;
        if (x.m_head.m_increment) {
            visit_expr_value(*x.m_head.m_increment);
            stepv = last_value;
        } else {
            stepv = emit_const_i32(1);
        }
        MLIR_ValueHandle next = emit_binop_i32(OP_TYPE_LLVM_ADD,
            str_lit("llvm.add"), iv, stepv);
        emit_store_i32(next, slot.ptr);
        emit_branch(header_b);

        cur_block = exit_b;
        block_terminated = false;
    }

    void visit_WhileLoop(const ASR::WhileLoop_t &x) {
        if (x.n_orelse > 0) {
            throw CodeGenError("mlir-new: while-loop else not supported",
                x.base.base.loc);
        }
        if (!x.m_test) {
            throw CodeGenError("mlir-new: while-loop missing test expression",
                x.base.base.loc);
        }

        MLIR_BlockHandle header_b = new_cfg_block();
        MLIR_BlockHandle body_b = new_cfg_block();
        MLIR_BlockHandle exit_b = new_cfg_block();
        emit_branch(header_b);

        cur_block = header_b;
        block_terminated = false;
        MLIR_ValueHandle cond = visit_expr_bool(*x.m_test);
        emit_cond_branch(cond, body_b, exit_b);

        cur_block = body_b;
        block_terminated = false;
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
            if (block_terminated) {
                break;
            }
        }
        if (!block_terminated) {
            emit_branch(header_b);
        }

        cur_block = exit_b;
        block_terminated = false;
    }

    void visit_stmt(ASR::stmt_t &s) {
        if (ASR::is_a<ASR::Assignment_t>(s)) {
            visit_Assignment(
                *ASR::down_cast<ASR::Assignment_t>(&s));
        } else if (ASR::is_a<ASR::Print_t>(s)) {
            visit_Print(*ASR::down_cast<ASR::Print_t>(&s));
        } else if (ASR::is_a<ASR::FileWrite_t>(s)) {
            visit_FileWrite(*ASR::down_cast<ASR::FileWrite_t>(&s));
        } else if (ASR::is_a<ASR::DoLoop_t>(s)) {
            visit_DoLoop(*ASR::down_cast<ASR::DoLoop_t>(&s));
        } else if (ASR::is_a<ASR::WhileLoop_t>(s)) {
            visit_WhileLoop(*ASR::down_cast<ASR::WhileLoop_t>(&s));
        } else {
            throw CodeGenError(
                "mlir-new: statement not supported yet", s.base.loc);
        }
    }

    void visit_Program(const ASR::Program_t &x) {
        emit_module_runtime_decls();

        fn_body_r = MLIR_CreateRegion(&ctx);
        cur_block = MLIR_CreateBlock(&ctx);
        MLIR_AppendRegionBlock(&ctx, fn_body_r, cur_block);
        block_terminated = false;

        entry_const_one = emit_const_i64(1);

        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Variable_t>(*item.second)) {
                emit_variable_storage(
                    *ASR::down_cast<ASR::Variable_t>(item.second));
            }
        }
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Variable_t>(*item.second)) {
                emit_variable_static_init(
                    *ASR::down_cast<ASR::Variable_t>(item.second));
            }
        }
        for (size_t i = 0; i < x.n_body; i++) {
            visit_stmt(*x.m_body[i]);
            if (block_terminated) {
                break;
            }
        }
        if (!block_terminated) {
            MLIR_ValueHandle zero = emit_const_i32(0);
            emit_return_i32(zero);
        }

        MLIR_TypeHandle fn_ty = MLIR_CreateTypeLLVMFunction(&ctx, i32_ty,
            nullptr, 0, false);
        append_module(emit_llvm_func(str_lit("main"), fn_ty, fn_body_r));
    }

    void visit_TranslationUnit(const ASR::TranslationUnit_t &x) {
        emit_module_skeleton();
        for (auto &item : x.m_symtab->get_scope()) {
            if (ASR::is_a<ASR::Program_t>(*item.second)) {
                visit_Program(
                    *ASR::down_cast<ASR::Program_t>(item.second));
                return;
            }
        }
        throw CodeGenError(
            "mlir-new: no program unit found (expected a single program)");
    }
};

Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
    ASR::asr_t &asr, diag::Diagnostics &diagnostics) {
    ensure_mlir_corec_platform_initialized();
    (void)al;
    if (!ASR::is_a<ASR::unit_t>(asr)) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: expected a translation unit (asr unit)",
            diag::Level::Error, diag::Stage::CodeGen));
        Error err;
        return err;
    }
    const ASR::unit_t *u = ASR::down_cast<ASR::unit_t>(&asr);
    if (u->type != ASR::unitType::TranslationUnit) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: expected unit type translation unit",
            diag::Level::Error, diag::Stage::CodeGen));
        Error err;
        return err;
    }
    const ASR::TranslationUnit_t *tu =
        ASR::down_cast<ASR::TranslationUnit_t>(u);

    LlvmEmitter emit;
    try {
        emit.visit_TranslationUnit(*tu);
    } catch (const CodeGenError &e) {
        diagnostics.diagnostics.push_back(e.d);
        if (emit.arena) {
            arena_destroy(emit.arena);
        }
        return Error();
    }

    std::string llvm_dialect_mlir;
    const char *debug_print = std::getenv("MLIR_NEW_DEBUG_PRINT");
    if (debug_print && debug_print[0] == '1' && debug_print[1] == '\0') {
        llvm_dialect_mlir = copy_mlir_string(
            MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));
    }

    const char *env = std::getenv("USE_MLIR_Upstream");
    bool use_upstream = env && env[0] == '1' && env[1] == '\0';

    string llvm_s = use_upstream
        ? MLIR_TranslateModuleToLLVMIRUpstream(&emit.ctx, emit.module_op)
        : MLIR_TranslateModuleToLLVMIR(&emit.ctx, emit.module_op);
    std::string llvm_ir = copy_mlir_string(llvm_s);
    if (llvm_ir.empty()) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: MLIR_TranslateModuleToLLVMIR returned empty "
            "output",
            diag::Level::Error, diag::Stage::CodeGen));
        arena_destroy(emit.arena);
        return Error();
    }

    arena_destroy(emit.arena);
    return Result<std::unique_ptr<MLIRModule>>(std::unique_ptr<MLIRModule>(
        new MLIRModule(std::move(llvm_ir), std::move(llvm_dialect_mlir))));
}

} // namespace LCompilers
