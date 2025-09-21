
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

#include "Utils/FuncBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
namespace mlir::utils {
FunctionCallBuilder::FunctionCallBuilder(StringRef functionName,
                                         FunctionType func_type)
    : function_name(functionName), function_type(func_type){};

FunctionCallBuilderResult FunctionCallBuilder::create(
    Location loc, OpBuilder &builder, ValueRange arguments) const {
  auto module = builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
  FunctionCallBuilderResult res;
  if (auto function = module.lookupSymbol<func::FuncOp>(function_name)) {
    res.function = function;
  } else {
    res.function = OpBuilder::atBlockEnd(module.getBody())
                       .create<func::FuncOp>(loc, function_name, function_type);
    res.func_created = true;
  }
  res.call = builder.create<func::CallOp>(loc, res.function, arguments);
  return res;
}
}  // namespace mlir::utils