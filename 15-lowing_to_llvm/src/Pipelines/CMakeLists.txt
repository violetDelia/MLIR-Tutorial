add_mlir_library(MLIRNorthStarPipelines${ch_num}
    NorthStarBasicPipeline.cpp

    DEPENDS

    LINK_LIBS PUBLIC
    MLIRIR
)
