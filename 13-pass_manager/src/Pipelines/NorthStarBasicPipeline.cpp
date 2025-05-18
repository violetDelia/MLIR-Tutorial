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

#include "Conversion/Passes.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Pipelines/Pipelines.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pipeline {
void buildBuffeNorthStarBasicPipeline(
    OpPassManager &pm, const NorthStarBasicPipelineOptions &options) {
  mlir::north_star::MarkDistributeParallelParametersPassOptions
      mark_distribute_parallel_option{.DPNums = options.DP_Nums, .TPNums = 1};
  pm.addPass(mlir::north_star::createMarkDistributeParallelParametersPass(
      mark_distribute_parallel_option));
  pm.addNestedPass<func::FuncOp>(
      mlir::north_star::createApplyDistributeTransformPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::north_star::createDeviceRegionFusionPass());
  pm.addPass(mlir::north_star::createConvertNorthStarToLinalgPass());
};

void registerNorthStarBasicPipelines() {
  PassPipelineRegistration<NorthStarBasicPipelineOptions>(
      "north-star-basic-pipeline", "basic pipeline ",
      buildBuffeNorthStarBasicPipeline);
};

}  // namespace mlir::pipeline
