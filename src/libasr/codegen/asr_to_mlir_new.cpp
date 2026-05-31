#include <libasr/codegen/asr_to_mlir_new.h>
#include <libasr/codegen/asr_to_asr_dialect.h>

#include <mutex>

extern "C" {
#include <platform/platform.h>
}

namespace LCompilers {

static std::once_flag mlir_corec_platform_init;

void ensure_mlir_corec_platform_initialized() {
    std::call_once(mlir_corec_platform_init,
                   []() { platform_init(0, nullptr); });
}

Result<std::unique_ptr<MLIRModule>> asr_to_mlir_new(Allocator &al,
    ASR::asr_t &asr, diag::Diagnostics &diagnostics) {
    ensure_mlir_corec_platform_initialized();
    return asr_to_asr_dialect(al, asr, diagnostics, AsrDialectPipelineStage::Full);
}

} // namespace LCompilers
