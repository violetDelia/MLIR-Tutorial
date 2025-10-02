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
#include <cstdint>
#include <memory>

#include "Dialect/NorthStar/IR/NorthStarAttrs.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/Key.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "device-region-fusion"

namespace mlir::north_star {
#define GEN_PASS_DEF_DEVICEREGIONFUSIONPASS
#include "Dialect/NorthStar/Transforms/Passes.h.inc"

}  // namespace mlir::north_star
using namespace ::mlir;
using namespace ::mlir::north_star;

namespace {

namespace {
static inline llvm::SmallString<4> getFusionName(
    mlir::ArrayRef<::mlir::Operation*> ops) {
  llvm::SmallString<4> name;
  for (auto op : ops) {
    name.append(op->getName().stripDialect());
    name.append("_");
    for (auto type : op->getOperandTypes()) {
      if (auto shaped = llvm::dyn_cast_or_null<ShapedType>(type)) {
        for (auto index : llvm::index_range(0, shaped.getRank())) {
          if (shaped.isDynamicDim(index)) {
            name.append("d_");
          } else {
            name.append(llvm::to_string(shaped.getDimSize(index)));
            name.append("_");
          }
        }
      }
    }
  }
  return name;
}

static inline int getDeviceid(mlir::ArrayRef<::mlir::Operation*> ops) {
  if (auto tensor = llvm::cast_or_null<north_star::NSTensorType>(
          ops.back()->getResultTypes().front())) {
    return tensor.getDeviceId();
  }
  llvm_unreachable("");
  return -1;
}

static inline llvm::MapVector<Value, std::pair<Operation*, int>>
getFusionInputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> opSet(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> res;
  for (auto op : ops) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand))
        res[operand] = std::make_pair(nullptr, 0);
      if (opSet.contains(operand.getDefiningOp())) continue;
      res[operand] = std::make_pair(op, index);
    }
  }
  return res;
}

static inline llvm::MapVector<Value, std::pair<Operation*, int>>
getFusionOutputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> opSet(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> outs;
  for (auto op : ops) {
    for (auto [index, res] : llvm::enumerate(op->getResults())) {
      for (auto user : res.getUsers()) {
        if (opSet.contains(user)) continue;
        outs[res] = std::make_pair(op, index);
        break;
      }
    }
  }
  return outs;
}
}  // namespace
void FusionOps(::mlir::RewriterBase& rewriter,
               mlir::ArrayRef<::mlir::Operation*> ops, ::mlir::Location loc) {
  if (ops.size() == 0) return;
  auto context = rewriter.getContext();
  auto insertPoint = rewriter.saveInsertionPoint();
  auto name = getFusionName(ops);
  auto deviceId = getDeviceid(ops);
  name.append(llvm::to_string(deviceId));
  auto inputsMap = getFusionInputs(ops);
  auto outputsMap = getFusionOutputs(ops);
  llvm::SmallVector<Value> inputsVal;
  llvm::SmallVector<Value> outputVal;
  llvm::SmallVector<Type> outputsType;
  llvm::SmallVector<Type> inputsType;
  for (auto [key, val] : inputsMap) {
    inputsVal.push_back(key);
    inputsType.push_back(key.getType());
  }
  for (auto [key, val] : outputsMap) {
    outputsType.push_back(key.getType());
  }
  rewriter.setInsertionPoint((*ops.begin())->getParentOp());
  auto kernel = rewriter.create<func::FuncOp>(
      loc, name, FunctionType::get(context, inputsType, outputsType));
  kernel->setAttr(KDeviceFunc, UnitAttr::get(context));
  auto block = kernel.addEntryBlock();
  std::map<Operation*, Operation*> opMap;
  for (auto op : ops) {
    auto cloneOp = op->clone();
    block->push_back(cloneOp);
    opMap[op] = cloneOp;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand)) continue;
      if (opMap.contains(operand.getDefiningOp())) {
        opMap[op]->setOperand(
            index,
            opMap[operand.getDefiningOp()]->getResult(
                llvm::cast_or_null<OpResult>(operand).getResultNumber()));
      }
    }
  }
  for (auto [key, val] : outputsMap) {
    outputVal.push_back(opMap[val.first]->getResult(val.second));
  }
  for (auto [index, key] : llvm::enumerate(inputsMap)) {
    opMap[key.second.first]->setOperand(key.second.second,
                                         block->getArgument(index));
  }

  rewriter.setInsertionPointToEnd(block);
  rewriter.create<func::ReturnOp>(loc, outputVal);
  rewriter.setInsertionPoint(insertPoint.getBlock(), insertPoint.getPoint());
  auto call = rewriter.create<func::CallOp>(loc, kernel, inputsVal);
  for (auto [index, key] : llvm::enumerate(outputsMap)) {
    rewriter.replaceAllUsesWith(key.first, call->getResult(index));
  }
  return;
}

struct BufferCastOpDeviceRegionFusion
    : public OpRewritePattern<::mlir::north_star::BufferCastOp> {
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult matchAndRewrite(::mlir::north_star::BufferCastOp op,
                                        PatternRewriter& rewriter) const {
    auto loc = op->getLoc();
    llvm::SmallVector<llvm::SetVector<Operation*>> op_list;
    for (auto res : op->getResults()) {
      rewriter.setInsertionPointAfterValue(res);
      llvm::SetVector<Operation*> ops;
      for (auto use : res.getUsers()) {
        addops(ops, use);
      }
      if (ops.size() != 0) op_list.push_back(ops);
    }
    if (op_list.size() == 0) return llvm::failure();
    for (auto ops : op_list) {
      if (DeviceKernelOp::FusionOps(rewriter, ops.takeVector(), loc).failed()) {
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv("fusion error!"));
        return llvm::failure();
      };
      // FusionOps(rewriter, ops.takeVector(), loc);
    }
    return llvm::success();
  }

  void addops(llvm::SetVector<Operation*>& ops, Operation* op) const {
    if (!isa<DistributeParallelOp>(op)) return;
    ops.insert(op);
    for (auto user : op->getUsers()) {
      addops(ops, user);
    }
  }
};

}  // namespace

void ::mlir::north_star::populateDeviceRegionFusionPatterns(
    RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.addWithLabel<BufferCastOpDeviceRegionFusion>(
      StringRef("BufferCastOpDeviceRegionFusion"), context, 100);
};

struct DeviceRegionFusionPass
    : ::mlir::north_star::impl::DeviceRegionFusionPassBase<
          DeviceRegionFusionPass> {
  using DeviceRegionFusionPassBase<
      DeviceRegionFusionPass>::DeviceRegionFusionPassBase;
  void runOnOperation() override;
};

void DeviceRegionFusionPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto module = getOperation();
  LLVM_DEBUG(
      llvm::dbgs() << llvm::formatv("root op: {0}\n", module->getName()));

  RewritePatternSet patterns(&getContext());
  ::mlir::north_star::populateDeviceRegionFusionPatterns(patterns);
  GreedyRewriteConfig config;
  bool changed;
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), FrozenRewritePatternSet(std::move(patterns)), config,
          &changed)))
    signalPassFailure();
  LLVM_DEBUG(
      llvm::dbgs() << llvm::formatv("region has changed: {0}\n", changed));
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}
