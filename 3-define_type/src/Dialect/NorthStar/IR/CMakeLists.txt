add_mlir_dialect_library(MLIRNorthStarDialect${ch_num}
    NorthStarDialect.cpp
    NorthStarTypes.cpp

    DEPENDS
    NorthStarDialectIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
)

