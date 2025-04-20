add_mlir_dialect_library(MLIRNorthStarDialect${ch_num}
    NorthStarDialect.cpp
    NorthStarTypes.cpp
    NorthStarAttrs.cpp
    NorthStarOps.cpp

    DEPENDS
    MLIRNorthStarDialectIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
    MLIRDistributeParallelismInterfaces${ch_num}
)

