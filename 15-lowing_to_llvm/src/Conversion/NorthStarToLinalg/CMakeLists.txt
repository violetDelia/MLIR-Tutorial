add_mlir_conversion_library(MLIRNorthStarToLinalg${ch_num}
  NorthStarToLinalg.cpp
  NorthStarToLinalgPass.cpp

  DEPENDS
  MLIRNorthStarConversionPassesIncGen${ch_num}

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRNorthStarDialect${ch_num}
  MLIRPass
  MLIRTransformUtils
  )

