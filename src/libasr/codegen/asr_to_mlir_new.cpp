#include <libasr/codegen/asr_to_mlir_new.h>
#include <libasr/codegen/asr_to_asr_dialect.h>
#include <libasr/codegen/evaluator.h>
#include <libasr/diagnostics.h>
#include <libasr/location.h>

#include <cstdlib>
#include <iostream>
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

static diag::Diagnostic make_asr_dialect_codegen_diagnostic(
        const char *prefix, const char *detail,
        uint32_t fortran_loc_first, uint32_t fortran_loc_last,
        diag::Level level) {
    std::string msg = std::string(prefix);
    if (detail && detail[0] != '\0') {
        msg += ": ";
        msg += detail;
    }
    if (fortran_loc_first != 0) {
        Location loc;
        loc.first = fortran_loc_first;
        loc.last = fortran_loc_last ? fortran_loc_last : fortran_loc_first;
        return diag::Diagnostic(msg, level, diag::Stage::CodeGen,
            {diag::Label("", {loc})});
    }
    return diag::Diagnostic(msg, level, diag::Stage::CodeGen);
}

static diag::Diagnostic make_asr_dialect_verify_diagnostic(
        const char *prefix, diag::Level level) {
    const ASR_DialectVerifyError *err = ASR_DialectGetLastVerifyError();
    const char *detail = (err && err->message[0] != '\0') ? err->message : NULL;
    uint32_t loc_first = err ? err->fortran_loc_first : 0;
    uint32_t loc_last = err ? err->fortran_loc_last : 0;
    return make_asr_dialect_codegen_diagnostic(
        prefix, detail, loc_first, loc_last, level);
}

static diag::Diagnostic make_asr_dialect_lowering_diagnostic(const char *prefix) {
    const ASR_DialectCodegenError *err = ASR_DialectGetLastCodegenError();
    const char *detail = (err && err->message[0] != '\0') ? err->message : NULL;
    uint32_t loc_first = err ? err->fortran_loc_first : 0;
    uint32_t loc_last = err ? err->fortran_loc_last : 0;
    return make_asr_dialect_codegen_diagnostic(
        prefix, detail, loc_first, loc_last, diag::Level::Error);
}

static bool mlir_new_pipeline_selected(
        const std::string &backend,
        const MlirShowOptions &show) {
    return backend == "mlir-new"
        || show.show_mlir_asr_dialect
        || show.show_mlir_high_dialect
        || show.show_mlir_llvm_dialect;
}

std::optional<MlirNewRequest> get_mlir_new_request(
        const std::string &backend,
        const MlirShowOptions &show) {
    if (!mlir_new_pipeline_selected(backend, show)) {
        return std::nullopt;
    }

    MlirNewRequest request{};
    const char *upstream = std::getenv("USE_MLIR_Upstream");
    request.use_upstream_lowering = upstream && upstream[0] == '1';

    if (show.show_mlir_asr_dialect) {
        request.target = MlirNewPipelineTarget::AsrDialect;
    } else if (show.show_mlir_high_dialect || show.show_mlir) {
        request.target = MlirNewPipelineTarget::HighMlir;
    } else if (show.show_mlir_llvm_dialect) {
        request.target = MlirNewPipelineTarget::LlvmDialect;
    } else if (show.show_llvm_from_mlir) {
        request.target = MlirNewPipelineTarget::LlvmIr;
    } else {
        request.target = MlirNewPipelineTarget::ObjectFile;
    }

    request.dump_only = request.target != MlirNewPipelineTarget::ObjectFile;
    return request;
}

void write_mlir_new_dump(std::ostream &out, MLIRModule &m,
        MlirNewPipelineTarget target) {
    switch (target) {
        case MlirNewPipelineTarget::AsrDialect:
            out << m.mlir_asr_dialect_dump();
            break;
        case MlirNewPipelineTarget::HighMlir:
            out << m.mlir_high_dialect_dump();
            break;
        case MlirNewPipelineTarget::LlvmDialect:
            out << m.mlir_llvm_dialect_dump();
            break;
        case MlirNewPipelineTarget::LlvmIr:
            out << m.llvm_str();
            break;
        case MlirNewPipelineTarget::ObjectFile:
            break;
    }
}

Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
    ASR::asr_t &asr, diag::Diagnostics &diagnostics,
    const MlirNewRequest &request) {
    ensure_mlir_corec_platform_initialized();
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
    emit.asr_al = &al;
    try {
        emit.visit_TranslationUnit(
            *ASR::down_cast<ASR::TranslationUnit_t>(u));
    } catch (const AsrDialectError &e) {
        diagnostics.diagnostics.push_back(e.d);
        cleanup_emit(emit);
        return Error();
    }

    MlirNewPipelineTarget target = request.target;
    if (!ASR_DialectVerify(&emit.ctx, emit.module_op)) {
        if (target == MlirNewPipelineTarget::AsrDialect) {
            diagnostics.diagnostics.push_back(
                make_asr_dialect_verify_diagnostic(
                    "asr_to_mlir_new: ASR dialect verification reported issues "
                    "(continuing in dump mode)",
                    diag::Level::Warning));
        } else {
            diagnostics.diagnostics.push_back(
                make_asr_dialect_verify_diagnostic(
                    "asr_to_mlir_new: ASR dialect verification failed",
                    diag::Level::Error));
            cleanup_emit(emit);
            return Error();
        }
    }


    if (target == MlirNewPipelineTarget::AsrDialect) {
        // Human-oriented debug dump only; not reparsable IR and not used by lowering.
        std::string asr_dialect_text =
            copy_mlir_string(ASR_DialectPrint(&emit.ctx, emit.module_op));
        cleanup_emit(emit);
        return make_mlir_module(std::move(asr_dialect_text),
            std::string(), std::string(), std::string());
    }

    ASR_DialectOptions opts{};
    opts.verify_asr_dialect = true;
    opts.allow_unimplemented_nodes = false;
    if (!ASR_DialectLowerToHighMLIR(&emit.ctx, emit.module_op, &opts)) {
        diagnostics.diagnostics.push_back(
            make_asr_dialect_lowering_diagnostic(
                "asr_to_mlir_new: ASR dialect lowering failed"));
        cleanup_emit(emit);
        return Error();
    }

    std::string high_mlir =
        copy_mlir_string(MLIR_PrintOperationUpstream(&emit.ctx, emit.module_op));

    if (target == MlirNewPipelineTarget::HighMlir) {
        cleanup_emit(emit);
        return make_mlir_module(std::string(), std::move(high_mlir),
            std::string(), std::string());
    }

    bool lowered = request.use_upstream_lowering
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
        return make_mlir_module(std::string(), std::move(high_mlir),
            std::move(llvm_dialect_mlir), std::string());
    }

    string llvm_s = request.use_upstream_lowering
        ? MLIR_TranslateModuleToLLVMIRUpstream(&emit.ctx, emit.module_op)
        : MLIR_TranslateModuleToLLVMIR(&emit.ctx, emit.module_op);
    std::string llvm_ir = copy_mlir_string(llvm_s);

    cleanup_emit(emit);
    return make_mlir_module(std::string(), std::move(high_mlir),
        std::move(llvm_dialect_mlir), std::move(llvm_ir));
}

} // namespace LCompilers
