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

#ifndef PIPELINES_PIPELINS_H
#define PIPELINES_PIPELINS_H
#include <cstdint>

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
namespace mlir::pipeline {

/// Options for the buffer deallocation pipeline.
struct NorthStarBasicPipelineOptions
    : public PassPipelineOptions<NorthStarBasicPipelineOptions> {
  PassOptions::Option<int64_t> DP_Nums{
      *this, "DP_Nums", llvm::cl::desc("数据并行参数."), llvm::cl::init(1)};
};

void buildNorthStarBasicPipeline(OpPassManager &pm,
                                 const NorthStarBasicPipelineOptions &options);

void registerNorthStarBasicPipelines();

void registerNorthStarBasicPipelinesExtennsion(mlir::DialectRegistry &registry);
}  // namespace mlir::pipeline

#endif  // PIPELINES_PIPELINS_H