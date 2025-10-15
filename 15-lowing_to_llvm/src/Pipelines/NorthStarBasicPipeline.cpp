//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#include <cstdlib>
#include <string>

#include "Conversion/NorthStarToFunc/NorthStarToFunc.h"
#include "Conversion/Passes.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Pipelines/Pipelines.h"
#include "mlir-c/IR.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MLProgram/Transforms/Passes.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Mesh/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/Transforms/Passes.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Unit.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
namespace {
void applyInterpreter(::mlir::OpPassManager &pm, const char *entry_point) {
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint = entry_point;
  pm.addPass(mlir::transform::createInterpreterPass(options));
}

void registerPasses() {
  mlir::registerConvertElementwiseToLinalgPass();
  mlir::registerLinalgElementwiseOpFusionPass();
  mlir::func::registerFuncBufferizePass();
  mlir::bufferization::registerFinalizingBufferizePass();
  mlir::bufferization::registerBufferDeallocationPass();
  mlir::bufferization::registerBufferHoistingPass();
  mlir::bufferization::registerBufferLoopHoisting();
  mlir::registerConvertVectorToSCF();
  mlir::registerSCFToControlFlow();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertIndexToLLVMPass();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerConvertVectorToLLVMPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::LLVM::registerLLVMLegalizeForExportPass();
  mlir::registerMem2RegPass();
  mlir::memref::registerExpandStridedMetadataPass();
  mlir::registerConvertAffineToStandardPass();
  mlir::registerReconcileUnrealizedCasts();
}

}  // namespace
namespace mlir::pipeline {
void buildBuffeNorthStarBasicPipeline(
    OpPassManager &pm, const NorthStarBasicPipelineOptions &options) {
  registerPasses();
  mlir::north_star::MarkDistributeParallelParametersPassOptions
      mark_distribute_parallel_option{.DPNums = options.DP_Nums, .TPNums = 1};
  pm.addPass(mlir::north_star::createMarkDistributeParallelParametersPass(
      mark_distribute_parallel_option));
  pm.addNestedPass<func::FuncOp>(
      mlir::north_star::createApplyDistributeTransformPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::north_star::createDeviceRegionFusionPass());
  pm.addPass(mlir::north_star::createEliminateBufferCastPass());
  pm.addPass(mlir::north_star::createConvertNorthStarToLinalgPass());
  pm.addPass(mlir::north_star::createNorthStarLegalizePass());
  pm.addPass(mlir::createCanonicalizerPass());
  mlir::SmallVector<std::string> transform_library_paths;

  transform_library_paths.push_back(std::getenv("NorthStarPassInclude"));
  mlir::transform::PreloadLibraryPassOptions preload_options{
      .transformLibraryPaths = transform_library_paths};

  pm.addPass(mlir::transform::createPreloadLibraryPass(preload_options));
  // applyInterpreter(pm, "linalg_analysis");
  applyInterpreter(pm, "linalg_decompose");
  
  pm.addPass(mlir::north_star::createConvertNorthStarToFuncPass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  mlir::bufferization::OneShotBufferizationOptions bufferization_options;
  bufferization_options.allowReturnAllocsFromLoops = true;
  bufferization_options.allowUnknownOps = true;
  bufferization_options.testAnalysisOnly = false;
  bufferization_options.bufferizeFunctionBoundaries = false;
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferization_options));
  applyInterpreter(pm, "linalg_basic_fuse");
  // applyInterpreter(pm, "finnal_bufferization");
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::affine::createLoopFusionPass());
  pm.addPass(mlir::createLoopInvariantSubsetHoistingPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  //     affine:: AffineVectorizeOptions vector_option;
  //       vector_option.vectorSizes = {4};
  //       vector_option.vectorizeReductions =false;
  //   pm.addPass(mlir::affine::createAffineVectorize(vector_option));
  pm.addPass(mlir::affine::createLoopFusionPass(0, 0, true));

  mlir::VectorTransferToSCFOptions vec_scf_options;
  vec_scf_options.unroll = true;
  pm.addPass(mlir::createConvertVectorToSCFPass(vec_scf_options));
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCSEPass());
  applyInterpreter(pm, "memref_basic_opt");
  applyInterpreter(pm, "lowing_to_llvm");
  applyInterpreter(pm, "llvm_basic_opt");
  pm.addPass(mlir::north_star::createNorthStarRuntimeCallPass());
};

void registerNorthStarBasicPipelines() {
  PassPipelineRegistration<NorthStarBasicPipelineOptions>(
      "north-star-basic-pipeline", "basic pipeline ",
      buildBuffeNorthStarBasicPipeline);
};
void registerNorthStarBasicPipelinesExtennsion(
    mlir::DialectRegistry &registry) {
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::bufferization::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerTransformDialectExtension(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::func::registerTransformDialectExtension(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::func::registerInlinerExtension(registry);
}

}  // namespace mlir::pipeline
