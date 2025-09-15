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

#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace {

struct GetTensorOpRefineResultPattern final
    : public OpConversionPattern<mlir::north_star::GetTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::GetTensorOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::GetTensorOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto convert = getTypeConverter();
    auto new_op = rewriter.create<north_star::GetTensorOp>(
        loc, convert->convertType(op.getType()), adaptor.getBuffer(),
        op.getDeviceId());
    rewriter.replaceOp(op, new_op);
  }
};
}  // namespace

namespace mlir::north_star {
namespace {

static Value materializeToNSTensor(OpBuilder &builder, NSTensorType type,
                                   ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<RankedTensorType>(inputs[0].getType()));
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(isa<NSTensorType>(inputs[0].getType()));
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

}  // namespace
void initNorthStarToFuncTypeConvert(TypeConverter &typeConverter) {
  typeConverter.addConversion([](NSTensorType type) {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addConversion([](BufferType type) { return type; });
  typeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1) return std::nullopt;

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1) return std::nullopt;

        return builder
            .create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
      });
}
void populateNorthStarToFuncPatterns(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  patterns.add<GetTensorOpRefineResultPattern>(typeConverter,
                                               patterns.getContext(), 1);
};
}  // namespace mlir::north_star
