add_mlir_conversion_library(MLIRNorthStarToFunc${ch_num}
  NorthStarToFunc.cpp
  NorthStarToFuncPass.cpp

  DEPENDS
  MLIRNorthStarConversionPassesIncGen${ch_num}

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRNorthStarDialect${ch_num}
  MLIRPass
  MLIRTransformUtils
  MLIRTutorialUtils${ch_num}
  )

