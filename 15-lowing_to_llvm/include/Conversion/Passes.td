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
  let summary = "lowing northstart to linalg";
  let description = [{
    "lowing northstart to linalg";
  }];
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::cf::ControlFlowDialect"
  ];
}

def NorthStarLegalizePass : Pass<"north-star-legalize","::mlir::ModuleOp"> {
  let summary = "legalize northstar for lowing to std";
  let description = [{
    "legalize northstar for lowing to std";
  }];
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
}

def ConvertNorthStarToFuncPass : Pass<"convert-north-satr-to-func","::mlir::ModuleOp"> {
  let summary = "lowing northstart to func";
  let description = [{
    "lowing northstart to func";
  }];
  let dependentDialects = [
    "::mlir::north_star::NorthStarDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect"
  ];
}



#endif  // CONVERSION_PASSES_TD