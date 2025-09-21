add_mlir_library(MLIRTutorialUtils${ch_num}
    File.cpp
    FuncBuilder.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/Utils

    DEPENDS

    LINK_LIBS PUBLIC
    MLIRIR
)

