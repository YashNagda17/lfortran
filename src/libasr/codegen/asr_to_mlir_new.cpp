#include <libasr/codegen/asr_to_mlir_new.h>
#include <libasr/codegen/asr_to_asr_dialect.h>
#include <libasr/codegen/evaluator.h>
#include <libasr/diagnostics.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>

extern "C" {
#include <asr_dialect_api.h>
#include <mlir_api.h>
#include <base/arena.h>
#include <platform/platform.h>
}

namespace LCompilers {

static std::once_flag mlir_corec_platform_init;

static void ensure_mlir_corec_platform_initialized() {
    std::call_once(mlir_corec_platform_init,
                   []() { platform_init(0, nullptr); });
}

static std::string copy_mlir_string(string s) {
    if (!s.str || s.size == 0) {
        return std::string();
    }
    return std::string(s.str, s.str + s.size);
}

static void cleanup_emit(ASRToAsrDialectVisitor &emit) {
    ASR_DialectModuleStorageClear(&emit.ctx);
    arena_destroy(emit.arena);
}

static std::unique_ptr<MLIRModule> make_mlir_module(
        std::string asr_dialect_text,
        std::string high_mlir,
        std::string llvm_dialect_mlir,
        std::string llvm_ir) {
    auto mod = std::make_unique<MLIRModule>(
        std::move(high_mlir), std::move(llvm_dialect_mlir), std::move(llvm_ir));
    mod->mlir_asr_dialect_text = std::move(asr_dialect_text);
    return mod;
}

Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
    ASR::asr_t &asr, diag::Diagnostics &diagnostics,
    MlirNewPipelineTarget target) {
    ensure_mlir_corec_platform_initialized();
    (void)al;
    if (!ASR::is_a<ASR::unit_t>(asr)) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: expected translation unit",
            diag::Level::Error, diag::Stage::CodeGen));
        return Error();
    }
    const ASR::unit_t *u = ASR::down_cast<ASR::unit_t>(&asr);
    if (u->type != ASR::unitType::TranslationUnit) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: expected unit type translation unit",
            diag::Level::Error, diag::Stage::CodeGen));
        return Error();
    }

    ASRToAsrDialectVisitor emit;
    try {
        emit.visit_TranslationUnit(
            *ASR::down_cast<ASR::TranslationUnit_t>(u));
    } catch (const AsrDialectError &e) {
        diagnostics.diagnostics.push_back(e.d);
        cleanup_emit(emit);
        return Error();
    }

    if (!ASR_DialectVerify(&emit.ctx, emit.module_op)) {
        if (target == MlirNewPipelineTarget::AsrDialect) {
            diagnostics.diagnostics.push_back(diag::Diagnostic(
                "asr_to_mlir_new: ASR dialect verification reported issues "
                "(continuing in dump mode)",
                diag::Level::Warning, diag::Stage::CodeGen));
        } else {
            diagnostics.diagnostics.push_back(diag::Diagnostic(
                "asr_to_mlir_new: ASR dialect verification failed",
                diag::Level::Error, diag::Stage::CodeGen));
            cleanup_emit(emit);
            return Error();
        }
    }

    std::string asr_dialect_text =
        copy_mlir_string(ASR_DialectPrint(&emit.ctx, emit.module_op));

    if (target == MlirNewPipelineTarget::AsrDialect) {
        cleanup_emit(emit);
        return make_mlir_module(std::move(asr_dialect_text),
            std::string(), std::string(), std::string());
    }

    ASR_DialectOptions opts{};
    opts.verify_asr_dialect = true;
    opts.allow_unimplemented_nodes = false;
    if (!ASR_DialectLowerToHighMLIR(&emit.ctx, emit.module_op, &opts)) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: ASR_DialectLowerToHighMLIR failed",
            diag::Level::Error, diag::Stage::CodeGen));
        cleanup_emit(emit);
        return Error();
    }

    std::string high_mlir =
        copy_mlir_string(MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));

    if (target == MlirNewPipelineTarget::HighMlir) {
        cleanup_emit(emit);
        return make_mlir_module(std::move(asr_dialect_text),
            std::move(high_mlir), std::string(), std::string());
    }

    bool use_upstream = std::getenv("USE_MLIR_Upstream") &&
        std::getenv("USE_MLIR_Upstream")[0] == '1';
    bool lowered = use_upstream
        ? MLIR_LowerToLLVMDialectUpstream(&emit.ctx, emit.module_op)
        : MLIR_LowerToLLVMDialect(&emit.ctx, emit.module_op);
    if (!lowered) {
        diagnostics.diagnostics.push_back(diag::Diagnostic(
            "asr_to_mlir_new: MLIR_LowerToLLVMDialect failed",
            diag::Level::Error, diag::Stage::CodeGen));
        cleanup_emit(emit);
        return Error();
    }

    std::string llvm_dialect_mlir =
        copy_mlir_string(MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));

    if (target == MlirNewPipelineTarget::LlvmDialect) {
        cleanup_emit(emit);
        return make_mlir_module(std::move(asr_dialect_text),
            std::move(high_mlir), std::move(llvm_dialect_mlir), std::string());
    }

    string llvm_s = use_upstream
        ? MLIR_TranslateModuleToLLVMIRUpstream(&emit.ctx, emit.module_op)
        : MLIR_TranslateModuleToLLVMIR(&emit.ctx, emit.module_op);
    std::string llvm_ir = copy_mlir_string(llvm_s);

    cleanup_emit(emit);
    return make_mlir_module(std::move(asr_dialect_text),
        std::move(high_mlir), std::move(llvm_dialect_mlir), std::move(llvm_ir));
}

} // namespace LCompilers
