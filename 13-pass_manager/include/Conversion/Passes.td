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

#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD
include "mlir/Pass/PassBase.td"

def ConvertNorthStarToLinalgPass : Pass<"convert-north-satr-to-linalg","::mlir::ModuleOp"> {
  let summary = "标记全局并行参数";
  let description = [{
    "标记全局并行参数。";
  }];
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}


#endif  // CONVERSION_PASSES_TD