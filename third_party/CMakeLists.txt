set(LLVM_ENABLE_PROJECTS "mlir;clang" CACHE STRING " " FORCE) 
set(LLVM_TARGETS_TO_BUILD "Native" CACHE STRING " " FORCE) 
add_subdirectory(llvm-project/llvm)