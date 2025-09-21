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
    res.call->setAttr(KDeviceIdAttr,rewriter.getI64IntegerAttr(op.getDeviceId()));
    if(res.func_created){
      res.function->setAttr(KDeviceFunc,rewriter.getUnitAttr());
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

}  // namespace

namespace mlir::north_star {
namespace {}  // namespace
void initNorthStarToFuncTypeConvert(TypeConverter &typeConverter) {
  typeConverter.addConversion([](NSTensorType type) {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addConversion([](BufferType type) { return type; });
  typeConverter.addConversion([](Type type) { return type; });
  // typeConverter.addSourceMaterialization(
  //     [&](OpBuilder &builder, Type resultType, ValueRange inputs,
  //         Location loc) -> std::optional<Value> {
  //       if (inputs.size() != 1) return std::nullopt;

  //       return builder
  //           .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
  //           .getResult(0);
  //     });
  // typeConverter.addTargetMaterialization(
  //     [&](OpBuilder &builder, Type resultType, ValueRange inputs,
  //         Location loc) -> std::optional<Value> {
  //       if (inputs.size() != 1) return std::nullopt;

  //       return builder
  //           .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
  //           .getResult(0);
  //     });
}
void populateNorthStarToFuncPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  patterns.add<DeviceKernelOpToFuncPattern, ReturnOpToFuncPattern>(
      typeConverter, patterns.getContext(), 1);
};
}  // namespace mlir::north_star
