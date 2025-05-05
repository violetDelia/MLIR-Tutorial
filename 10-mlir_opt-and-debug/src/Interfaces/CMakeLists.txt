set(LLVM_OPTIONAL_SOURCES
    DistributeParallelismInterfaces.cpp
    FusionRegionInterfaces.cpp
)

add_mlir_library(MLIRDistributeParallelismInterfaces${ch_num}
    DistributeParallelismInterfaces.cpp

    DEPENDS
    MLIRDistributeParallelismInterfacesIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
)

add_mlir_library(MLIRFusionRegionInterfaces${ch_num}
    FusionRegionInterfaces.cpp

    DEPENDS
    MLIRFusionRegionInterfacesIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
)
