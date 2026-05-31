#ifndef LFORTRAN_ASR_TO_ASR_DIALECT_H
#define LFORTRAN_ASR_TO_ASR_DIALECT_H

#include <libasr/asr.h>
#include <libasr/codegen/evaluator.h>

namespace LCompilers {

    enum class AsrDialectPipelineStage {
        DialectOnly,
        Full
    };

    Result<std::unique_ptr<MLIRModule>> asr_to_asr_dialect(Allocator &al,
        ASR::asr_t &asr, diag::Diagnostics &diagnostics,
        AsrDialectPipelineStage stage = AsrDialectPipelineStage::Full);

} // namespace LCompilers

#endif
