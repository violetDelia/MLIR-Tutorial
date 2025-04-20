add_mlir_library(MLIRDistributeParallelismInterfaces${ch_num}
    DistributeParallelismInterfaces.cpp

    DEPENDS
    MLIRDistributeParallelismInterfacesIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
)

