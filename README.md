# `2_lfortran` — MLIR-new integration

This directory holds an experimental LFortran tree that adds a **translate-only** MLIR backend (`--backend=mlir-new`). Fortran is lowered to **LLVM-dialect MLIR** via the vendored C API under [`lfortran/src/mlir/`](lfortran/src/mlir/), then translated to **LLVM IR** and linked through the existing LLVM object-code path.

Primary [`lfortran/`](../lfortran/) (repo root sibling) uses high-level MLIR + a lowering pass. **`2_lfortran/lfortran` intentionally does not** — there is no `func`/`memref`/`arith` stage and no `MLIR_LowerToLLVMDialect`.

## Problem / motivation

LFortran’s classic `--backend=mlir` builds MLIR through the C++ MLIR API. The certik-style C API in [`2_lfortran/mlir/`](../mlir/) offers a smaller, testable surface for emitting MLIR and translating it to LLVM IR. The goal here is to:

1. Vendor only the **translation** slice of that API inside LFortran.
2. Emit **LLVM dialect ops directly from ASR** (no high-level MLIR module).
3. Keep **`USE_MLIR_Upstream`** env switching for native vs upstream translation.
4. Use **hosted corec platform fixes** from primary `lfortran` (avoid bare-metal `memcpy` shims crashing glibc-linked binaries).

## Pipeline

```
Fortran → ASR (default passes) → asr_to_mlir_new.cpp
       → llvm-dialect MLIR (llvm.func, llvm.alloca, llvm.gep, llvm.add, …)
       → MLIR_TranslateModuleToLLVMIR[*Upstream]
       → LLVM IR text → llvm::Module → object file / dumps
```

Implementation map: [`Implementation2.md`](../Implementation2.md).

## Recent codegen updates (`asr_to_mlir_new.cpp`)

The mlir-new visitor now supports (via **LLVM dialect API only**):

| Feature | ASR | LLVM dialect ops |
|---------|-----|------------------|
| Integer locals | `Variable` | `llvm.alloca`, `llvm.load` / `llvm.store` |
| Fixed-size integer arrays | `Variable` + `Array` | `llvm.alloca` of `!llvm.array<N x i32>`, `llvm.getelementptr` |
| Array element read/write | `ArrayItem` | `llvm.gep`, `llvm.load` / `llvm.store` |
| Array constant init | `ArrayConstant` | element-wise `llvm.store` |
| Integer arithmetic | `IntegerBinOp` | `llvm.add`, `llvm.sub`, `llvm.mul`, `llvm.sdiv` |
| `do` loops | `DoLoop` | `llvm.br`, `llvm.cond_br`, `llvm.icmp` |
| Print integer | `Print` / `write(*,*)` | `llvm.call @printf` |

Still **not** supported: allocatable/pointer arrays, rank > 1 in general, reals, functions/subroutines, I/O beyond default unit, whole-array assignment, `do` `else` branches.

### Example program

```fortran
program arrays_loops
  implicit none
  integer :: a(5), i, s
  a = (/ 1, 2, 3, 4, 5 /)
  s = 0
  do i = 1, 5
     s = s + a(i)
  end do
  print *, s
end program
```

## Repository layout

| Path | Role |
|------|------|
| [`lfortran/`](lfortran/) | LFortran source with mlir-new wired in |
| [`lfortran/src/mlir/`](lfortran/src/mlir/) | Vendored MLIR C API + corec (translate-only) |
| [`lfortran/src/libasr/codegen/asr_to_mlir_new.cpp`](lfortran/src/libasr/codegen/asr_to_mlir_new.cpp) | ASR → LLVM dialect emitter |
| [`mlir/`](../mlir/) | Upstream certik MLIR C API (source for vendored files) |
| [`Implementation2.md`](../Implementation2.md) | Detailed handoff / file list / verification |

## Build options

Requires **conda env `mlir19`** (LLVM 19 + MLIR), per project [`mlir_build.txt`](../mlir_build.txt) and [`AGENTS.md`](../AGENTS.md).

```bash
conda activate mlir19
cd 2_lfortran/lfortran

./build0.sh
cmake . -GNinja \
  -DWITH_LLVM=yes \
  -DWITH_MLIR=yes \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
./build1.sh
```

| CMake flag | Effect |
|------------|--------|
| `-DWITH_LLVM=yes` | LLVM backend (required for mlir-new object emission) |
| `-DWITH_MLIR=yes` | Builds `lfortran_mlir_c_api`, enables `--backend=mlir-new` |
| `-DWITH_MLIR=no` | Classic LFortran only; mlir-new flags error at runtime |

**Environment**

| Variable | Effect |
|----------|--------|
| `USE_MLIR_Upstream=1` | Use `MLIR_TranslateModuleToLLVMIRUpstream` |
| `USE_MLIR_Upstream=0` or unset | Use native `MLIR_TranslateModuleToLLVMIR` |

## How to run / test

Binary (after build): `2_lfortran/lfortran/src/bin/lfortran`

### Backends

```bash
# Classic LLVM IR (no MLIR)
./src/bin/lfortran --show-llvm program.f90

# Classic MLIR (C++ MLIR path)
./src/bin/lfortran --backend=mlir --show-mlir program.f90

# MLIR-new: LLVM dialect → LLVM IR
./src/bin/lfortran --backend=mlir-new --show-llvm-from-mlir program.f90

# MLIR-new: dump LLVM IR via --show-mlir
./src/bin/lfortran --backend=mlir-new --show-mlir program.f90

# MLIR-new: dump llvm-dialect MLIR (debug)
./src/bin/lfortran --backend=mlir-new --show-mlir-llvm-dialect program.f90

# Compile to object
./src/bin/lfortran --backend=mlir-new -c program.f90 -o program.o
```

### Compare translators

```bash
USE_MLIR_Upstream=0 ./src/bin/lfortran --backend=mlir-new --show-llvm-from-mlir program.f90
USE_MLIR_Upstream=1 ./src/bin/lfortran --backend=mlir-new --show-llvm-from-mlir program.f90
```

### Smoke tests from repo root

```bash
# Simple scalar (see a.f90)
./src/bin/lfortran --backend=mlir-new --show-mlir ../a.f90

# Platform / corec: cpp backend should not segfault on -c
./src/bin/lfortran --backend=cpp -c ../a.f90 -o /tmp/a.o
```

## Related trees

- **`lfortran/`** (top-level): primary integration; high-level MLIR + lowering pass.
- **`2_lfortran/mlir/`**: standalone MLIR C API development / tests.
- **`2_lfortran/lfortran/`**: LFortran fork with slim vendored API and direct LLVM-dialect emission.

See also [`.claude/memory/mlir_new_implementation.md`](../.claude/memory/mlir_new_implementation.md) for agent-oriented notes.
