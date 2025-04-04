add_mlir_dialect_library(MLIRNorthStarDialect${ch_num}
    NorthStarDialect.cpp
    NorthStarTypes.cpp
    NorthStarAttrs.cpp
    NorthStarOps.cpp
    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/Dialect/NorthStar

    DEPENDS
    NorthStarDialectIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
    MLIRDistributeParallelismInterfaces${ch_num}
)

