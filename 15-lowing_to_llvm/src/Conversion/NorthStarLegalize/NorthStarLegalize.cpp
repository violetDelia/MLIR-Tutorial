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

#include "Conversion/NorthStarLegalize/NorthStarLegalize.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/FuncBuilder.h"
#include "Utils/Key.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace {
struct DeviceKernelOpToFuncPattern final
    : public OpConversionPattern<mlir::north_star::DeviceKernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::DeviceKernelOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::DeviceKernelOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto func_type =
        rewriter.getFunctionType(op->getOperandTypes(), op->getResultTypes());
    std::string func_name = op.getSymName().str();
    auto func_builder = mlir::utils::FunctionCallBuilder(func_name, func_type);
    auto res = func_builder.create(loc, rewriter, op->getOperands());
    if (res.func_created) {
      rewriter.cloneRegionBefore(op.getRegion(), res.function.getRegion(),
                                 res.function.getRegion().end());
    }
    res.call->setAttr(KDeviceIdAttr,
                      rewriter.getI64IntegerAttr(op.getDeviceId()));
    res.call->setAttr(KDeviceFunc, rewriter.getUnitAttr());
    if (res.func_created) {
      res.function->setAttr(KDeviceFunc, rewriter.getUnitAttr());
    }
    rewriter.replaceOp(op, res.call);
  }
};

struct ReturnOpToFuncPattern final
    : public OpConversionPattern<mlir::north_star::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::ReturnOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::ReturnOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
  }
};

struct FuncReturnOpRerewriterPattern final
    : public OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(func::ReturnOp op) const final { return llvm::success(); }
  void rewrite(func::ReturnOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
  }
};

struct FuncFuncOpRerewriterPattern final
    : public OpConversionPattern<mlir::func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(func::FuncOp op) const final { return llvm::success(); }
  void rewrite(func::FuncOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto new_func_type = llvm::dyn_cast_or_null<FunctionType>(
        getTypeConverter()->convertType(op.getFunctionType()));
    if (!new_func_type) {
      return;
    }
    auto new_op =
        rewriter.create<func::FuncOp>(loc, op.getSymName(), new_func_type);
    rewriter.cloneRegionBefore(op.getBody(), new_op.getBody(),
                               new_op.getBody().end());
    rewriter.setInsertionPointToStart(&new_op.getFunctionBody().front());
    for (auto arg : new_op.getFunctionBody().front().getArguments()) {
      if (auto ns_tensor =
              llvm::dyn_cast_or_null<north_star::NSTensorType>(arg.getType())) {
        arg.setType(RankedTensorType::get(ns_tensor.getShape(),
                                          ns_tensor.getElementType()));
        new_op.setArgAttr(arg.getArgNumber(), KFuncDeviceIdAttr,
                          rewriter.getI64IntegerAttr(ns_tensor.getDeviceId()));
        auto to_ns_tensor = rewriter.create<north_star::TensorToNSTensorOp>(
            loc, ns_tensor, arg, ns_tensor.getDeviceId());
        rewriter.replaceAllUsesExcept(arg, to_ns_tensor, to_ns_tensor);
      }
    }
    rewriter.replaceOp(op, new_op);
  }
};

}  // namespace

namespace mlir::north_star {
namespace {}  // namespace
void initNorthStarLegalizeTypeConvert(TypeConverter &type_converter) {
  type_converter.addConversion([](NSTensorType type) {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  type_converter.addConversion([](BufferType type) { return type; });
  type_converter.addConversion([](RankedTensorType type) { return type; });
  type_converter.addConversion([](FunctionType type) {
    SmallVector<Type> inputs;
    SmallVector<Type> outputs;
    for (auto type : type.getInputs()) {
      if (auto ns_tensor = llvm::dyn_cast_or_null<NSTensorType>(type)) {
        inputs.push_back(RankedTensorType::get(ns_tensor.getShape(),
                                               ns_tensor.getElementType()));
      } else {
        inputs.push_back(type);
      }
    }
    for (auto type : type.getResults()) {
      if (auto ns_tensor = llvm::dyn_cast_or_null<NSTensorType>(type)) {
        outputs.push_back(RankedTensorType::get(ns_tensor.getShape(),
                                                ns_tensor.getElementType()));
      } else {
        outputs.push_back(type);
      }
    }
    return FunctionType::get(type.getContext(), inputs, outputs);
  });

  auto materialize_cast = [](OpBuilder &builder, Type type, ValueRange inputs,
                            Location loc) -> std::optional<Value> {
    if (inputs.size() != 1) return std::nullopt;

    if (auto ns_tensor =
            llvm::dyn_cast_or_null<NSTensorType>(inputs[0].getType())) {
      if (auto tensor = llvm::dyn_cast_or_null<RankedTensorType>(type)) {
        return builder.create<NSTensorToTensorOp>(loc, type, inputs[0],
                                                  ns_tensor.getDeviceId());
      }
    }
    if (auto tensor =
            llvm::dyn_cast_or_null<RankedTensorType>(inputs[0].getType())) {
      if (auto ns_tensor = llvm::dyn_cast_or_null<NSTensorType>(type)) {
        return builder.create<TensorToNSTensorOp>(loc, type, inputs[0],
                                                  ns_tensor.getDeviceId());
      }
    }
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
        .getResult(0);
  };

  type_converter.addSourceMaterialization(materialize_cast);
  type_converter.addTargetMaterialization(materialize_cast);
  type_converter.addArgumentMaterialization(materialize_cast);
}

void populateNorthStarLegalizePatterns(TypeConverter &type_converter,
                                       RewritePatternSet &patterns) {
  patterns.add<DeviceKernelOpToFuncPattern, ReturnOpToFuncPattern,
               FuncReturnOpRerewriterPattern, FuncFuncOpRerewriterPattern>(
      type_converter, patterns.getContext());
};
}  // namespace mlir::north_star
