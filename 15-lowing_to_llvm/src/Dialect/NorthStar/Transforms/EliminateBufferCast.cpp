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

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/Key.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "eliminate-buffercast"

namespace mlir::north_star {
#define GEN_PASS_DEF_ELIMINATEBUFFERCASTPASS
#include "Dialect/NorthStar/Transforms/Passes.h.inc"

}  // namespace mlir::north_star
using namespace ::mlir;
using namespace ::mlir::north_star;

namespace {
struct BufferCastOpToCommunicationPattern final
    : public OpRewritePattern<mlir::north_star::BufferCastOp> {
  using OpRewritePattern::OpRewritePattern;
  ;

  LogicalResult match(north_star::BufferCastOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::BufferCastOp op,
               PatternRewriter& rewriter) const final {
    llvm::SmallVector<Type> new_results;
    auto inputsNums = op->getNumOperands();
    auto outputsNums = op->getNumResults();
    auto distributeAttr = op.getDistributeAttr();
    if (auto DPAttr = llvm::cast_or_null<DataParallelAttr>(distributeAttr)) {
      if (inputsNums == 1) {
        rewriteToScatter(op, rewriter);
        return;
      }
      if (outputsNums == 1) {
        rewriteToGather(op, rewriter);
        return;
      }
      llvm::llvm_unreachable_internal("not impl", __FILE__, __LINE__);
    }
    llvm::llvm_unreachable_internal("not impl", __FILE__, __LINE__);
  };

  void rewriteToScatter(north_star::BufferCastOp op,
                        PatternRewriter& rewriter) const {
    auto loc = op->getLoc();
    auto context = op->getContext();
    auto inBuffer = rewriter.create<north_star::BufferOp>(loc, op.getInputs());
    llvm::SmallVector<Value> outMemories;
    for (auto type : op->getResultTypes()) {
      auto tensorType = llvm::cast_or_null<north_star::NSTensorType>(type);
      if (!tensorType)
        llvm::llvm_unreachable_internal("unexpect type", __FILE__, __LINE__);
      if (!tensorType.hasStaticShape())
        llvm::llvm_unreachable_internal("not impl", __FILE__, __LINE__);
      auto tensor = rewriter.create<tensor::EmptyOp>(
          loc, tensorType.getShape(), tensorType.getElementType());
      tensor->setAttr(KDeviceIdAttr,
                      rewriter.getI64IntegerAttr(tensorType.getDeviceId()));
      auto NSTensor = rewriter.create<north_star::TensorToNSTensorOp>(
          loc, tensorType, tensor, tensorType.getDeviceId());
      outMemories.push_back(NSTensor);
    }
    auto outBuffer = rewriter.create<north_star::BufferOp>(loc, outMemories);
    auto scatter =
        rewriter.create<north_star::ScatterOp>(loc, inBuffer, outBuffer);
    llvm::SmallVector<Value> output;
    for (auto [index, res, device] :
         llvm::enumerate(op.getOutputs(), op.outputDevices())) {
      auto memory = rewriter.create<north_star::GetTensorOp>(loc, res.getType(),
                                                             outBuffer, device);
      output.push_back(memory);
    }
    rewriter.replaceAllOpUsesWith(op, output);
  }
  void rewriteToGather(north_star::BufferCastOp op,
                       PatternRewriter& rewriter) const {
    auto loc = op->getLoc();
    auto context = op->getContext();
    auto inBuffer = rewriter.create<north_star::BufferOp>(loc, op.getInputs());
    llvm::SmallVector<Value> outMemories;
    for (auto type : op->getResultTypes()) {
      auto tensorType = llvm::cast_or_null<north_star::NSTensorType>(type);
      if (!tensorType)
        llvm::llvm_unreachable_internal("unexpect type", __FILE__, __LINE__);
      if (!tensorType.hasStaticShape())
        llvm::llvm_unreachable_internal("not impl", __FILE__, __LINE__);
      auto tensor = rewriter.create<tensor::EmptyOp>(
          loc, tensorType.getShape(), tensorType.getElementType());
      tensor->setAttr(KDeviceIdAttr,
                      rewriter.getI64IntegerAttr(tensorType.getDeviceId()));
      auto NSTensor = rewriter.create<north_star::TensorToNSTensorOp>(
          loc, tensorType, tensor, tensorType.getDeviceId());
      outMemories.push_back(NSTensor);
    }
    auto outBuffer = rewriter.create<north_star::BufferOp>(loc, outMemories);
    auto scatter =
        rewriter.create<north_star::GatherOp>(loc, inBuffer, outBuffer);
    llvm::SmallVector<Value> output;
    for (auto [index, res, device] :
         llvm::enumerate(op.getOutputs(), op.outputDevices())) {
      auto memory = rewriter.create<north_star::GetTensorOp>(loc, res.getType(),
                                                             outBuffer, device);
      output.push_back(memory);
    }
    rewriter.replaceAllOpUsesWith(op, output);
  }
};

}  // namespace

void ::mlir::north_star::populateEliminateBufferCastPatterns(
    RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.addWithLabel<BufferCastOpToCommunicationPattern>(
      StringRef("BufferCastOpToCommunicationPattern"), context);
};

struct EliminateBufferCastPass
    : ::mlir::north_star::impl::EliminateBufferCastPassBase<
          EliminateBufferCastPass> {
  using EliminateBufferCastPassBase<
      EliminateBufferCastPass>::EliminateBufferCastPassBase;
  void runOnOperation() override;
};
void EliminateBufferCastPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  RewritePatternSet patterns(&getContext());
  populateEliminateBufferCastPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), FrozenRewritePatternSet(std::move(patterns)))))
    signalPassFailure();
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}
