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

#include <string>

#include "Conversion/NorthStarToFunc/NorthStarToFunc.h"
#include "Conversion/Passes.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Pipelines/Pipelines.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
namespace {
void applyInterpreter(::mlir::OpPassManager &pm, const char *entry_point) {
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint = entry_point;
  pm.addPass(mlir::transform::createInterpreterPass(options));
}
}  // namespace
namespace mlir::pipeline {
void buildBuffeNorthStarBasicPipeline(
    OpPassManager &pm, const NorthStarBasicPipelineOptions &options) {
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
  transform_library_paths.push_back(
      "/home/lfr/MLIR_Tutorial/linalg_include.mlir");
  mlir::transform::PreloadLibraryPassOptions preload_options{
      .transformLibraryPaths = transform_library_paths};

  pm.addPass(mlir::transform::createPreloadLibraryPass(preload_options));
  //applyInterpreter(pm, "linalg_analysis");
  applyInterpreter(pm, "linalg_decompose");
  pm.addPass(mlir::north_star::createConvertNorthStarToFuncPass());
  mlir::bufferization::OneShotBufferizationOptions bufferization_options;
  bufferization_options.allowReturnAllocsFromLoops = true;
  bufferization_options.allowUnknownOps = true;
  bufferization_options.testAnalysisOnly = false;
  bufferization_options.bufferizeFunctionBoundaries = false;
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferization_options));
  
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
}

}  // namespace mlir::pipeline
