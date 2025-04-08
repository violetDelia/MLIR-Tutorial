set(LLVM_TARGET_DEFINITIONS DistributeParallelismInterfaces.td)
mlir_tablegen(DistributeParallelismOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(DistributeParallelismOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(DistributeParallelismAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(DistributeParallelismAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRDistributeParallelismInterfacesIncGen${ch_num})

