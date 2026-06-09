#ifndef LFORTRAN_ASR_TO_MLIR_NEW_H
#define LFORTRAN_ASR_TO_MLIR_NEW_H

#include <libasr/asr.h>
#include <libasr/asr_utils.h>
#include <libasr/codegen/evaluator.h>

#include <iosfwd>
#include <optional>
#include <string>

namespace LCompilers {

    // How far the mlir-new pipeline runs before returning (like --show-asr vs
    // --show-llvm stopping at different compiler stages).
    enum class MlirNewPipelineTarget {
        AsrDialect,
        HighMlir,
        LlvmDialect,
        LlvmIr,
        ObjectFile,
    };

    struct MlirShowOptions {
        bool show_mlir_asr_dialect = false;
        bool show_mlir_high_dialect = false;
        bool show_mlir_llvm_dialect = false;
        bool show_mlir = false;
        bool show_llvm_from_mlir = false;
    };

    struct MlirNewRequest {
        MlirNewPipelineTarget target = MlirNewPipelineTarget::ObjectFile;
        bool use_upstream_lowering = false;
        bool dump_only = false;
    };

    // Returns a mlir-new request when the backend or show flags select the new
    // pipeline; otherwise std::nullopt (use classic MLIR lowering).
    std::optional<MlirNewRequest> get_mlir_new_request(
        const std::string &backend,
        const MlirShowOptions &show);

    void write_mlir_new_dump(std::ostream &out, MLIRModule &m,
        MlirNewPipelineTarget target);

    Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
        ASR::asr_t &asr, diag::Diagnostics &diagnostics,
        const MlirNewRequest &request);

} // namespace LCompilers

#endif // LFORTRAN_ASR_TO_MLIR_NEW_H
