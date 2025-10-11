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

#include <string>

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/Key.h"
#include "iostream"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define DEBUG_TYPE "apply-distribute-transform"

namespace mlir::north_star {
#define GEN_PASS_DEF_NORTHSTARRUNTIMECALLPASS
#include "Dialect/NorthStar/Transforms/Passes.h.inc"

}  // namespace mlir::north_star
using namespace ::mlir;
using namespace ::mlir::north_star;

struct MemrefCopyPattern final : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match(LLVM::CallOp op) const {
    if (op.getCallee()->str() != KMemrefCopyName) return llvm::failure();
    return llvm::success();
  }

  void rewrite(LLVM::CallOp op, PatternRewriter& rewriter) const final {
    llvm::SmallVector<Type> types{
        op.getCalleeFunctionType().getParams().begin(),
        op.getCalleeFunctionType().getParams().end()};
    types.push_back(LLVM::LLVMPointerType::get(op->getContext()));
    mlir::FunctionCallBuilder builder(
        KMemrefCopyBuiltinName, op.getCalleeFunctionType().getReturnType(),
        types);
    llvm::SmallVector<Value> args{op.getOperands().begin(),
                                  op->getOperands().end()};
    args.push_back(rewriter.create<north_star::GetHostStreamOp>(op->getLoc()));
    auto new_call = builder.create(op->getLoc(), rewriter, args);
    rewriter.replaceOp(op, new_call);
  };
};

struct FreePattern final : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match(LLVM::CallOp op) const {
    if (op.getCallee()->str() != KFreeName) return llvm::failure();
    return llvm::success();
  }

  void rewrite(LLVM::CallOp op, PatternRewriter& rewriter) const final {
    llvm::SmallVector<Type> types{
        op.getCalleeFunctionType().getParams().begin(),
        op.getCalleeFunctionType().getParams().end()};
    types.push_back(LLVM::LLVMPointerType::get(op->getContext()));
    mlir::FunctionCallBuilder builder(
        KFreeBuiltinName, op.getCalleeFunctionType().getReturnType(), types);
    llvm::SmallVector<Value> args{op.getOperands().begin(),
                                  op->getOperands().end()};
    args.push_back(rewriter.create<north_star::GetHostStreamOp>(op->getLoc()));
    auto new_call = builder.create(op->getLoc(), rewriter, args);
    rewriter.replaceOp(op, new_call);
  };
};

struct MallocPattern final : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match(LLVM::CallOp op) const {
    if (op.getCallee()->str() != KMallocName) return llvm::failure();
    return llvm::success();
  }

  void rewrite(LLVM::CallOp op, PatternRewriter& rewriter) const final {
    llvm::SmallVector<Type> types{
        op.getCalleeFunctionType().getParams().begin(),
        op.getCalleeFunctionType().getParams().end()};
    types.push_back(LLVM::LLVMPointerType::get(op->getContext()));
    mlir::FunctionCallBuilder builder(
        KMallocBuiltinName, op.getCalleeFunctionType().getReturnType(), types);
    llvm::SmallVector<Value> args{op.getOperands().begin(),
                                  op->getOperands().end()};
    args.push_back(rewriter.create<north_star::GetHostStreamOp>(op->getLoc()));
    auto new_call = builder.create(op->getLoc(), rewriter, args);
    rewriter.replaceOp(op, new_call);
  };
};

struct MemcpyPattern final : public OpRewritePattern<LLVM::MemcpyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult match(LLVM::MemcpyOp op) const { return llvm::success(); }

  void rewrite(LLVM::MemcpyOp op, PatternRewriter& rewriter) const final {
    llvm::SmallVector<Type> types{op->getOperandTypes().begin(),
                                  op.getOperandTypes().end()};
    types.push_back(rewriter.getI1Type());
    types.push_back(LLVM::LLVMPointerType::get(op->getContext()));
    mlir::FunctionCallBuilder builder(
        KMemcpyBuiltinName, LLVM::LLVMVoidType::get(op->getContext()), types);
    llvm::SmallVector<Value> args{op.getOperands().begin(),
                                  op->getOperands().end()};
    args.push_back(rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI1Type(), op.getIsVolatile()));
    args.push_back(rewriter.create<north_star::GetHostStreamOp>(op->getLoc()));
    auto new_call = builder.create(op->getLoc(), rewriter, args);
    op->erase();
  };
};

void ::mlir::north_star::populateNorthStarRuntimeCallPatterns(
    RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.addWithLabel<MemrefCopyPattern>(StringRef("MemrefCopyPattern"),
                                           context);
  patterns.addWithLabel<FreePattern>(StringRef("FreePattern"), context);
  patterns.addWithLabel<MallocPattern>(StringRef("MallocPattern"), context);
  patterns.addWithLabel<MemcpyPattern>(StringRef("MemcpyPattern"), context);
}

struct NorthStarRuntimeCallPass
    : ::mlir::north_star::impl::NorthStarRuntimeCallPassBase<
          NorthStarRuntimeCallPass> {
  using NorthStarRuntimeCallPassBase<
      NorthStarRuntimeCallPass>::NorthStarRuntimeCallPassBase;
  void runOnOperation() override;
};
void NorthStarRuntimeCallPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  RewritePatternSet patterns(&getContext());
  populateNorthStarRuntimeCallPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), FrozenRewritePatternSet(std::move(patterns)))))
    signalPassFailure();
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("root op: {0}\n",
                                           getOperation()->getName()));
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}
