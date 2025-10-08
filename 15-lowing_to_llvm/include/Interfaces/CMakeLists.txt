set(LLVM_TARGET_DEFINITIONS DistributeParallelismInterfaces.td)
mlir_tablegen(DistributeParallelismOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(DistributeParallelismOpInterfaces.cpp.inc -gen-op-interface-defs)
mlir_tablegen(DistributeParallelismAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(DistributeParallelismAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRDistributeParallelismInterfacesIncGen${ch_num})

set(LLVM_TARGET_DEFINITIONS FusionRegionInterfaces.td)
mlir_tablegen(FusionRegionOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(FusionRegionOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRFusionRegionInterfacesIncGen${ch_num})

set(LLVM_TARGET_DEFINITIONS BuiltinFunctionInterfaces.td)
mlir_tablegen(BuiltinFunctionOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(BuiltinFunctionOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRBuiltinFunctionInterfacesIncGen${ch_num})

