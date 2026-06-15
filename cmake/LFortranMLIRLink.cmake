# Set LFORTRAN_MLIR_LINK_LIBS for the MLIR C API (mlir_api_impl_upstream) and
# existing asr_to_mlir.cpp. Prefers linking all static libMLIR*.a archives next
# to LLVM (same layout as conda-forge / typical LLVM builds).

set(LFORTRAN_MLIR_LIB_SEARCH_DIRS "")
if (DEFINED LLVM_LIBRARY_DIR)
    list(APPEND LFORTRAN_MLIR_LIB_SEARCH_DIRS "${LLVM_LIBRARY_DIR}")
endif()
if (DEFINED MLIR_LIBRARY_DIR)
    list(APPEND LFORTRAN_MLIR_LIB_SEARCH_DIRS "${MLIR_LIBRARY_DIR}")
elseif (DEFINED MLIR_DIR)
    # MLIR_DIR is typically <prefix>/lib/cmake/mlir
    get_filename_component(_mlir_cmake_parent "${MLIR_DIR}" DIRECTORY)
    get_filename_component(_mlir_prefix_lib "${_mlir_cmake_parent}" DIRECTORY)
    list(APPEND LFORTRAN_MLIR_LIB_SEARCH_DIRS "${_mlir_prefix_lib}")
endif()
list(REMOVE_DUPLICATES LFORTRAN_MLIR_LIB_SEARCH_DIRS)

set(LFORTRAN_MLIR_ARCHIVE_LIBS "")
foreach (d IN LISTS LFORTRAN_MLIR_LIB_SEARCH_DIRS)
    if (EXISTS "${d}")
        file(GLOB _mlir_a "${d}/libMLIR*.a")
        list(APPEND LFORTRAN_MLIR_ARCHIVE_LIBS ${_mlir_a})
    endif()
endforeach()
list(REMOVE_DUPLICATES LFORTRAN_MLIR_ARCHIVE_LIBS)

set(LFORTRAN_MLIR_LINK_LIBS "")
if (LFORTRAN_MLIR_ARCHIVE_LIBS)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
        list(APPEND LFORTRAN_MLIR_LINK_LIBS "-Wl,--start-group")
    endif()
    list(APPEND LFORTRAN_MLIR_LINK_LIBS ${LFORTRAN_MLIR_ARCHIVE_LIBS})
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
        list(APPEND LFORTRAN_MLIR_LINK_LIBS "-Wl,--end-group")
    endif()
else()
    message(WARNING
        "No libMLIR*.a archives found under LLVM/MLIR library dirs. "
        "Falling back to a small target list; linking mlir_api_impl_upstream may fail. "
        "Use an LLVM+MLIR build that installs static MLIR libraries.")
    list(APPEND LFORTRAN_MLIR_LINK_LIBS
        MLIRIR
        MLIRLLVMToLLVMIRTranslation
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMDialect
        MLIROpenMPToLLVMIRTranslation
        MLIROpenMPDialect
    )
endif()
