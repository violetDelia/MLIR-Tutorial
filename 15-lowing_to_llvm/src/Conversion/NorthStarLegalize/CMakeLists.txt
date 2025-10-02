add_mlir_conversion_library(MLIRNorthStarLegalize${ch_num}
  NorthStarLegalize.cpp
  NorthStarLegalizePass.cpp

  DEPENDS
  MLIRNorthStarConversionPassesIncGen${ch_num}

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRNorthStarDialect${ch_num}
  MLIRPass
  MLIRTransformUtils
  MLIRTutorialUtils${ch_num}
  )

