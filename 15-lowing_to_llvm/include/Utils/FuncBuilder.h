
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

#ifndef UTILS_MLIR_UTILS_FUNCBUILDER_H
#define UTILS_MLIR_UTILS_FUNCBUILDER_H
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir::utils {
struct FunctionCallBuilderResult {
  func::CallOp call;
  func::FuncOp function;
  llvm::SmallVector<Value> replecement_results;
  bool func_created = false;
};

struct AutoCastOption {
  bool castTensor = false;
  bool castMemRef = false;
};

struct FunctionCallBuilder {
  FunctionCallBuilder(std::string functionName, FunctionType func_type);
  FunctionCallBuilder(std::string functionName, FunctionType func_type,
                      AutoCastOption auto_cast_option);

  FunctionCallBuilderResult create(Location loc, OpBuilder &builder,
                                   ValueRange arguments) const;

  std::string function_name;
  FunctionType function_type;
  AutoCastOption auto_cast_option;
};
}  // namespace mlir::utils
#endif  // UTILS_MLIR_UTILS_FUNCBUILDER_H