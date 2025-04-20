add_mlir_library(MLIRDistributeParallelismInterfaces${ch_num}
    DistributeParallelismInterfaces.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../Interfaces

    DEPENDS
    MLIRDistributeParallelismInterfacesIncGen${ch_num}

    LINK_LIBS PUBLIC
    MLIRIR
)

