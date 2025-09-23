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

#include "Conversion/NorthStarToFunc/NorthStarToFunc.h"

#include <cstdint>
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
    llvm::SmallVector<Value> updatedOperands =
        llvm::to_vector<4>(adaptor.getOperands());

    for (auto [index, operand] : llvm::enumerate(updatedOperands)) {
      if (auto ns_tensor = llvm::dyn_cast_or_null<north_star::NSTensorType>(
              operand.getType())) {
        Value new_operand = rewriter.create<north_star::NSTensorToTensorOp>(
            op.getLoc(), typeConverter->convertType(ns_tensor), operand,
            ns_tensor.getDeviceId());
        updatedOperands[index] = new_operand;
      }
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, updatedOperands);
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
    TypeConverter::SignatureConversion conversion(new_func_type.getNumInputs());
    rewriter.cloneRegionBefore(op.getBody(), new_op.getBody(),
                               new_op.getBody().end());
    auto new_block = rewriter.applySignatureConversion(
        &new_op.getFunctionBody().front(), conversion);
    new_block->dump();
    // if
    // (llvm::failed(rewriter.applySignatureConversion(&new_op.getFunctionBody().front(),
    //                                              getTypeConverter()))) {
    //   return;
    // };
    new_op->dump();
    rewriter.replaceOp(op, new_op);
  }
};

}  // namespace

namespace mlir::north_star {
namespace {}  // namespace
void initNorthStarToFuncTypeConvert(TypeConverter &typeConverter) {
  typeConverter.addConversion([](NSTensorType type) {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addConversion([](BufferType type) { return type; });
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([](FunctionType type) {
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
  auto materializeCast = [](OpBuilder &builder, Type type, ValueRange inputs,
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

  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);
  typeConverter.addArgumentMaterialization(materializeCast);
}

void populateNorthStarToFuncPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  patterns.add<DeviceKernelOpToFuncPattern, ReturnOpToFuncPattern,
               FuncReturnOpRerewriterPattern, FuncFuncOpRerewriterPattern>(
      typeConverter, patterns.getContext());
};
}  // namespace mlir::north_star
