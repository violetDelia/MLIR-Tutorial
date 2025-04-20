
add_mlir_dialect_library(MLIRNorthStarTransforms${ch_num}
  ApplyDistributeTransform.cpp
  MarkDistributeParallelParameters.cpp

  DEPENDS
  MLIRNorthStarPassesIncGen${ch_num}

  LINK_LIBS PUBLIC
  MLIRNorthStarDialect${ch_num}
)
