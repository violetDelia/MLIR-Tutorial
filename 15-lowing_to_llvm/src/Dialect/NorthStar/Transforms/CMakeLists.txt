
add_mlir_dialect_library(MLIRNorthStarTransforms${ch_num}
  ApplyDistributeTransform.cpp
  MarkDistributeParallelParameters.cpp
  DeviceRegionFusion.cpp
  DeviceRegionOutlining.cpp
  EliminateBufferCast.cpp
  NorthStarRuntimeCall.cpp

  DEPENDS
  MLIRNorthStarPassesIncGen${ch_num}

  LINK_LIBS PUBLIC
  MLIRNorthStarDialect${ch_num}
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRGPUToGPURuntimeTransforms
)
