
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

#include <algorithm>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace {

mlir::LogicalResult argmentCheck(mlir::FunctionType func_type,
                                 mlir::ValueRange arguments) {
  if (func_type.getNumInputs() != arguments.size()) {
    return mlir::failure();
  }
  for (auto [arg_type, func_arg_type] :
       llvm::zip(arguments.getTypes(), func_type.getInputs())) {
    if (arg_type != func_arg_type) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

static inline bool needAutoCast(
    mlir::Type type, const mlir::utils::AutoCastOption &auto_cast_option) {
  if (auto tensor_type = llvm::dyn_cast_or_null<mlir::TensorType>(type)) {
    if (auto_cast_option.castTensor) {
      return true;
    }
    return false;
  }
  return false;
}

static inline mlir::Type AutoCast(
    mlir::Type type, mlir::utils::AutoCastOption auto_cast_option) {
  if (auto tensor_type = llvm::dyn_cast_or_null<mlir::TensorType>(type)) {
    if (auto_cast_option.castTensor) {
      return mlir::UnrankedTensorType::get(tensor_type.getElementType());
    }
    return type;
  }
  return type;
}

static mlir::FunctionType AutoCastFunctionType(
    mlir::FunctionType func_type,
    const mlir::utils::AutoCastOption &auto_cast_option) {
  llvm::SmallVector<mlir::Type> new_input_types;
  std::transform(func_type.getInputs().begin(), func_type.getInputs().end(),
                 std::back_inserter(new_input_types), [&](mlir::Type type) {
                   return AutoCast(type, auto_cast_option);
                 });
  llvm::SmallVector<mlir::Type> new_result_types;
  std::transform(func_type.getResults().begin(), func_type.getResults().end(),
                 std::back_inserter(new_result_types), [&](mlir::Type type) {
                   return AutoCast(type, auto_cast_option);
                 });
  return mlir::FunctionType::get(func_type.getContext(), new_input_types,
                                 new_result_types);
}

static inline mlir::Value AutoCastArg(
    mlir::Value value, mlir::OpBuilder &builder, mlir::Location loc,
    const mlir::utils::AutoCastOption &auto_cast_option) {
  if (llvm::isa<mlir::TensorType>(value.getType())) {
    builder.setInsertionPointAfterValue(value);
    return builder.create<mlir::tensor::CastOp>(
        loc, AutoCast(value.getType(), auto_cast_option), value);
  }

  return value;
}

static llvm::LogicalResult AutoCastArg(
    mlir::ValueRange arguments, mlir::OpBuilder &builder, mlir::Location loc,
    const mlir::utils::AutoCastOption &auto_cast_option,
    llvm::SmallVector<mlir::Value> &new_args) {
  std::transform(arguments.begin(), arguments.end(),
                 std::back_inserter(new_args), [&](mlir::Value value) {
                   return needAutoCast(value.getType(), auto_cast_option)
                              ? AutoCastArg(value, builder, loc,
                                            auto_cast_option)
                              : value;
                 });
  return llvm::success();
}

}  // namespace
namespace mlir::utils {

FunctionCallBuilder::FunctionCallBuilder(std::string functionName,
                                         FunctionType func_type)
    : function_name(functionName), function_type(func_type){};

FunctionCallBuilder::FunctionCallBuilder(std::string functionName,
                                         FunctionType func_type,
                                         AutoCastOption auto_cast_option)
    : function_name(functionName),
      function_type(func_type),
      auto_cast_option(auto_cast_option){};

FunctionCallBuilderResult FunctionCallBuilder::create(
    Location loc, OpBuilder &builder, ValueRange arguments) const {
  if (argmentCheck(function_type, arguments).failed()) {
    llvm::errs() << "\n";
    llvm::report_fatal_error("function call argument check failed");
    return {};
  }
  auto module = builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
  FunctionCallBuilderResult res;
  if (auto function = module.lookupSymbol<func::FuncOp>(function_name)) {
    res.function = function;
  } else {
    res.function =
        OpBuilder::atBlockEnd(module.getBody())
            .create<func::FuncOp>(
                loc, function_name,
                AutoCastFunctionType(function_type, auto_cast_option));
    res.function.setPrivate();
    res.func_created = true;
  }
  ImplicitLocOpBuilder b(loc, builder);
  llvm::SmallVector<mlir::Value> new_arguments;
  if (AutoCastArg(arguments, b, loc, auto_cast_option, new_arguments)
          .failed()) {
    llvm::errs() << "\n";
    llvm::report_fatal_error("function call argument auto cast failed");
    return {};
  }
  res.call = builder.create<func::CallOp>(loc, res.function, new_arguments);
  // TODO: handle auto cast for return value
  for (auto [index, ret_type, call_ret_type] :
       llvm::enumerate(function_type.getResults(), res.call.getResultTypes())) {
    if (ret_type != call_ret_type) {
      if (isa<mlir::UnrankedTensorType>(call_ret_type) &&
          auto_cast_option.castTensor) {
        b.setInsertionPointAfterValue(res.call->getResult(index));
        auto cast_op = builder.create<mlir::tensor::CastOp>(
            loc, ret_type, res.call.getResult(index));
        res.replecement_results.push_back(cast_op);
      } else {
        b.setInsertionPointAfterValue(res.call->getResult(index));
        auto cast_op = builder.create<mlir::UnrealizedConversionCastOp>(
            loc, ret_type, ValueRange{res.call.getResult(index)});
        res.replecement_results.push_back(cast_op.getResult(0));
      }

    } else {
      res.replecement_results.push_back(res.call.getResult(index));
    }
  }
  return res;
}
}  // namespace mlir::utils