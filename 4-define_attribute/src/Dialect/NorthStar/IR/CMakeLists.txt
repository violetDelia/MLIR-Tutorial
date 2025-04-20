add_mlir_dialect_library(MLIRNorthStarDialect${ch_num}
    NorthStarDialect.cpp
    NorthStarTypes.cpp
    NorthStarAttrs.cpp

    DEPENDS
    NorthStarDialectIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
)

