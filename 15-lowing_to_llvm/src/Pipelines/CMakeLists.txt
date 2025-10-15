add_mlir_library(MLIRNorthStarPipelines${ch_num}
    NorthStarBasicPipeline.cpp

    DEPENDS

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRReconcileUnrealizedCasts
    MLIRBufferizationTransforms
    MLIRArithTransforms
    MLIRLinalgTransforms
    MLIRTensorTransforms
    MLIRTransformDialectTransforms
    MLIRLinalgTransformOps
    MLIRVectorTransformOps
    MLIRSCFTransformOps
    MLIRControlFlowDialect
    MLIRBufferizationTransformOps
    MLIRTensorTransformOps
    MLIRBufferizationPipelines
    MLIRTensorToLinalg
    MLIRMemRefTransforms
    MLIRAffineToStandard
    MLIRMemRefTransformOps
    MLIRIndexToLLVM
    MLIRArithToLLVM
    MLIRFuncTransformOps
    MLIRControlFlowToLLVM
    MLIRMathToLLVM
    MLIRMemRefToLLVM
    MLIRVectorToLLVM
    MLIRVectorToLLVMPass
    MLIRVectorToSCF
    MLIRFuncInlinerExtension
    MLIRNorthStarDialect${ch_num}
    MLIRNorthStarTransforms${ch_num}
    MLIRTutorialUtils${ch_num}
    MLIRNorthStarToLinalg${ch_num}
    MLIRNorthStarToFunc${ch_num}
    MLIRNorthStarLegalize${ch_num}

)
