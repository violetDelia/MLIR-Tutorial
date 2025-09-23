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

#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#define __USE_hasCanonicalizeMethod__ false

namespace mlir::north_star {

#if __USE_hasCanonicalizeMethod__

LogicalResult BufferCastOp::canonicalize(BufferCastOp op,
                                         PatternRewriter &rewriter) {
  // patterns and rewrites go here.
  Operation *above_cast = nullptr;
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    if (isa<BlockArgument>(operand)) return llvm::failure();
    if (!above_cast) {
      above_cast = operand.getDefiningOp();
    } else {
      if (operand.getDefiningOp() != above_cast) return llvm::failure();
    }
    if (operand.getType() != above_cast->getResult(index).getType())
      return llvm::failure();
    if (!above_cast->getResult(index).hasOneUse()) return llvm::failure();
  }
  above_cast = op->getOperand(0).getDefiningOp();
  for (auto [index, res] : llvm::enumerate(op->getResults())) {
    rewriter.replaceAllUsesWith(res, above_cast->getOperand(index));
  }
  rewriter.eraseOp(op);
  rewriter.eraseOp(above_cast);
  return llvm::success();
}

#else

namespace {
struct BufferCastOpFold
    : public OpRewritePattern< ::mlir::north_star::BufferCastOp> {
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult match(::mlir::north_star::BufferCastOp op) const {
    Operation *above_cast = nullptr;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand)) return llvm::failure();
      if (!above_cast) {
        above_cast = operand.getDefiningOp();
      } else {
        if (operand.getDefiningOp() != above_cast) return llvm::failure();
      }
      if (operand.getType() != above_cast->getResult(index).getType())
        return llvm::failure();
      if (!above_cast->getResult(index).hasOneUse()) return llvm::failure();
    }
    return llvm::success();
  }

  virtual void rewrite(::mlir::north_star::BufferCastOp op,
                       PatternRewriter &rewriter) const {
    Operation *above_cast = op->getOperand(0).getDefiningOp();
    for (auto [index, res] : llvm::enumerate(op->getResults())) {
      rewriter.replaceAllUsesWith(res, above_cast->getOperand(index));
    }
    rewriter.eraseOp(op);
    rewriter.eraseOp(above_cast);
  }
};

struct TensorToNSTensorOpEliminate
    : public OpRewritePattern< ::mlir::north_star::TensorToNSTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult match(::mlir::north_star::TensorToNSTensorOp op) const {
    if (isa<BlockArgument>(op.getInput())) return llvm::failure();
    if (auto ns_tensor_to_tensor = llvm::dyn_cast_or_null<NSTensorToTensorOp>(
            op.getInput().getDefiningOp())) {
      if (ns_tensor_to_tensor.getDeviceId() == op.getDeviceId())
        return llvm::success();
    }
    return llvm::failure();
  }

  virtual void rewrite(::mlir::north_star::TensorToNSTensorOp op,
                       PatternRewriter &rewriter) const {
    Operation *above_cast = op->getOperand(0).getDefiningOp();
    rewriter.replaceOp(op, above_cast->getOperand(0));
  }
};

struct NSTensorToTensorOpEliminate
    : public OpRewritePattern< ::mlir::north_star::NSTensorToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult match(::mlir::north_star::NSTensorToTensorOp op) const {
    if (isa<BlockArgument>(op.getInput())) return llvm::failure();
    if (auto ns_tensor_to_tensor = llvm::dyn_cast_or_null<TensorToNSTensorOp>(
            op.getInput().getDefiningOp())) {
      if (ns_tensor_to_tensor.getDeviceId() == op.getDeviceId())
        return llvm::success();
    }
    return llvm::failure();
  }

  virtual void rewrite(::mlir::north_star::NSTensorToTensorOp op,
                       PatternRewriter &rewriter) const {
    Operation *above_cast = op->getOperand(0).getDefiningOp();
    rewriter.replaceOp(op, above_cast->getOperand(0));
  }
};

}  // namespace

void mlir::north_star::BufferCastOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.addWithLabel<BufferCastOpFold>(StringRef("BufferCastOpFold"), context,
                                         100);
}
void mlir::north_star::TensorToNSTensorOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.addWithLabel<TensorToNSTensorOpEliminate>(
      StringRef("TensorToNSTensorOpEliminate"), context, 100);
};

void mlir::north_star::NSTensorToTensorOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.addWithLabel<NSTensorToTensorOpEliminate>(
      StringRef("NSTensorToTensorOpEliminate"), context, 100);
};

#endif
#undef __USE_hasCanonicalizeMethod__

namespace {
static bool isSplatZero(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
  }
  if (llvm::isa<IntegerType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
  }
  return false;
}

static bool isSplatOne(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType)) {
    return val && val.isSplat() &&
           (val.getSplatValue<APFloat>().convertToDouble() == 1);
  }
  if (llvm::isa<IntegerType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APInt>().isAllOnes();
  }
  return false;
}

DenseElementsAttr splatDenseBinaryFolder(
    DenseElementsAttr lhs, DenseElementsAttr rhs, ShapedType returnTy,
    function_ref<APInt(llvm::APInt, llvm::APInt)> int_calculate,
    function_ref<APFloat(llvm::APFloat, llvm::APFloat)> float_calculate) {
  if (rhs && lhs && rhs.isSplat() && lhs.isSplat()) {
    auto lhs_ele_type = llvm::cast<ShapedType>(lhs.getType()).getElementType();
    auto rhs_ele_type = llvm::cast<ShapedType>(rhs.getType()).getElementType();
    if (lhs_ele_type != rhs_ele_type) return {};
    if (llvm::isa<IntegerType>(lhs_ele_type)) {
      APInt l = lhs.getSplatValue<APInt>();
      APInt r = rhs.getSplatValue<APInt>();
      auto result = int_calculate(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }
    if (llvm::isa<FloatType>(lhs_ele_type)) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      auto result = float_calculate(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }
  }
  return {};
}
}  // namespace

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto res_type = getType();
  if (!isa<NSTensorType>(res_type)) return {};
  if (isa<ShapedType>(res_type)) {
    auto lhs_type = llvm::dyn_cast<ShapedType>(getLhs().getType());
    auto rhs_type = llvm::dyn_cast<ShapedType>(getRhs().getType());
    auto result_type = llvm::dyn_cast<ShapedType>(getType());
    if (!lhs_type.getElementType().isIntOrIndexOrFloat() ||
        !rhs_type.getElementType().isIntOrIndexOrFloat())
      return {};
    auto lhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getLhs());
    auto rhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getRhs());
    // add(x, 0) -> x
    if (lhs_type == result_type &&
        isSplatZero(result_type.getElementType(), rhs_attr))
      return getLhs();
    // add(0, x) -> x
    if (rhs_type == result_type &&
        isSplatZero(result_type.getElementType(), lhs_attr))
      return getRhs();
    if (!lhs_attr || !rhs_attr) return {};
    return splatDenseBinaryFolder(
        lhs_attr, rhs_attr, result_type,
        [](const APInt &a, const APInt &b) { return a + b; },
        [](const APFloat &a, const APFloat &b) { return a + b; });
  }
  return {};
}
}  // namespace mlir::north_star
