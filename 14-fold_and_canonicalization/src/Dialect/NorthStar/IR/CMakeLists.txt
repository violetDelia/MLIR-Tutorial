add_mlir_dialect_library(MLIRNorthStarDialect${ch_num}
    NorthStarDialect.cpp
    NorthStarTypes.cpp
    NorthStarAttrs.cpp
    NorthStarOps.cpp
    NorthStarCanonicalize.cpp

    DEPENDS
    MLIRNorthStarDialectIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
    MLIRDistributeParallelismInterfaces${ch_num}
    MLIRFusionRegionInterfaces${ch_num}
)

