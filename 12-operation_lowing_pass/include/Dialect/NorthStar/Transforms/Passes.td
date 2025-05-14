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
//

#ifndef DIALECT_NORTH_STAR_TRANSFORMS_PASSES_TD
#define DIALECT_NORTH_STAR_TRANSFORMS_PASSES_TD
include "mlir/Pass/PassBase.td"

def MarkDistributeParallelParametersPass : Pass<"mark-distribute-parallel-parameters","::mlir::ModuleOp"> {
  let summary = "标记全局并行参数";
  let description = [{
    "标记全局并行参数。";
  }];
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"DPNums", "DP", "std::int64_t", /*default=*/"1", "DPNums des">,
    Option<"TPNums", "TP", "std::int64_t", /*default=*/"1", "TPNums des">
  ];
  let statistics = [
    Statistic<"EPNums", "ep-nums", "Number of EP">
  ];
}

def ApplyDistributeTransformPass : Pass<"apply-distribute-transform","::mlir::func::FuncOp"> {
  let summary = "根据标记的并行参数进行变换";
  let description = [{
    "根据标记的并行参数进行变换。"
  }];
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect"
  ];
  let constructor = "mlir::north_star::createApplyDistributeTransformPass()";
}


def DeviceRegionFusionPass : Pass<"device-region-fusion","::mlir::func::FuncOp"> {
  let summary = "相同device的计算合并";
  let description = [{
    "相同device的计算合并"
  }];
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

#endif  // DIALECT_NORTH_STAR_TRANSFORMS_PASSES_TD