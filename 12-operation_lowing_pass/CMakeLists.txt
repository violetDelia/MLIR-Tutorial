set(ch_num 12)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
set(TUTORIA_TOOL_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/Tools/NS-opt)
set(TUTORIA_NS_OPT ${TUTORIA_TOOL_DIR}/NS-opt${ch_num})
add_subdirectory(include)
add_subdirectory(src)
add_executable(CH-${ch_num} "main.cpp")
target_link_libraries(CH-${ch_num}
    MLIRNorthStarDialect${ch_num}
    MLIRNorthStarTransforms${ch_num}
    MLIRTutorialUtils${ch_num}
    MLIRGPUDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRLinalgDialect
    MLIRParser
    MLIRPass
)
add_subdirectory(test)
