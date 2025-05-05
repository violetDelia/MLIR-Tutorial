//    Copyright 2024 时光丶人爱

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
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
  mlir::SetVector<Operation*> op_set(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> res;
  for (auto op : ops) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand))
        res[operand] = std::make_pair(nullptr, 0);
      if (op_set.contains(operand.getDefiningOp())) continue;
      res[operand] = std::make_pair(op, index);
    }
  }
  return res;
}

static inline llvm::MapVector<Value, std::pair<Operation*, int>>
getFusionOutputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> op_set(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> outs;
  for (auto op : ops) {
    for (auto [index, res] : llvm::enumerate(op->getResults())) {
      for (auto user : res.getUsers()) {
        if (op_set.contains(user)) continue;
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
  auto insert_point = rewriter.saveInsertionPoint();
  auto name = getFusionName(ops);
  auto device_id = getDeviceid(ops);
  name.append(llvm::to_string(device_id));
  auto inputs_map = getFusionInputs(ops);
  auto outputs_map = getFusionOutputs(ops);
  llvm::SmallVector<Value> inputs_val;
  llvm::SmallVector<Value> output_val;
  llvm::SmallVector<Type> outputs_type;
  llvm::SmallVector<Type> inputs_type;
  for (auto [key, val] : inputs_map) {
    inputs_val.push_back(key);
    inputs_type.push_back(key.getType());
  }
  for (auto [key, val] : outputs_map) {
    outputs_type.push_back(key.getType());
  }
  rewriter.setInsertionPoint((*ops.begin())->getParentOp());
  auto kernel = rewriter.create<func::FuncOp>(
      loc, name, FunctionType::get(context, inputs_type, outputs_type));
  kernel->setAttr(KDeviceFunc, UnitAttr::get(context));
  auto block = kernel.addEntryBlock();
  std::map<Operation*, Operation*> op_map;
  for (auto op : ops) {
    auto clone_op = op->clone();
    block->push_back(clone_op);
    op_map[op] = clone_op;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand)) continue;
      if (op_map.contains(operand.getDefiningOp())) {
        op_map[op]->setOperand(
            index,
            op_map[operand.getDefiningOp()]->getResult(
                llvm::cast_or_null<OpResult>(operand).getResultNumber()));
      }
    }
  }
  for (auto [key, val] : outputs_map) {
    output_val.push_back(op_map[val.first]->getResult(val.second));
  }
  for (auto [index, key] : llvm::enumerate(inputs_map)) {
    op_map[key.second.first]->setOperand(key.second.second,
                                         block->getArgument(index));
  }

  rewriter.setInsertionPointToEnd(block);
  rewriter.create<func::ReturnOp>(loc, output_val);
  rewriter.setInsertionPoint(insert_point.getBlock(), insert_point.getPoint());
  auto call = rewriter.create<func::CallOp>(loc, kernel, inputs_val);
  for (auto [index, key] : llvm::enumerate(outputs_map)) {
    rewriter.replaceAllUsesWith(key.first, call->getResult(index));
  }
  return;
}

struct BufferCastOpDeviceRegionFusion
    : public OpRewritePattern<::mlir::north_star::BufferCastOp> {
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult matchAndRewrite(::mlir::north_star::BufferCastOp op,
                                        PatternRewriter& rewriter) const {
    llvm::outs() << "match:" << getDebugName() << "\n";
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
      FusionOps(rewriter, ops.takeVector(), loc);
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

struct BufferCastOpFold
    : public OpRewritePattern<::mlir::north_star::BufferCastOp> {
  using OpRewritePattern::OpRewritePattern;

  virtual LogicalResult match(::mlir::north_star::BufferCastOp op) const {
    llvm::outs() << "match:" << getDebugName() << "\n";
    Operation* above_cast = nullptr;
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
                       PatternRewriter& rewriter) const {
    Operation* above_cast = op->getOperand(0).getDefiningOp();
    for (auto [index, res] : llvm::enumerate(op->getResults())) {
      rewriter.replaceAllUsesWith(res, above_cast->getOperand(index));
    }
    rewriter.eraseOp(op);
    rewriter.eraseOp(above_cast);
    llvm::outs() << "match:" << getDebugName() << "\n";
  }
};
}  // namespace

void ::mlir::north_star::populateDeviceRegionFusionPatterns(
    RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.addWithLabel<BufferCastOpDeviceRegionFusion>(
      StringRef("BufferCastOpDeviceRegionFusion"), context, 100);
};

void ::mlir::north_star::populateBufferCastOpCanonicalizationPatterns(
    RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.addWithLabel<BufferCastOpFold>(StringRef("BufferCastOpFold"),
                                          context, 2);
}

struct DeviceRegionFusionPass
    : ::mlir::north_star::impl::DeviceRegionFusionPassBase<
          DeviceRegionFusionPass> {
  using DeviceRegionFusionPassBase<
      DeviceRegionFusionPass>::DeviceRegionFusionPassBase;
  void runOnOperation() override;
};

void DeviceRegionFusionPass::runOnOperation() {
  llvm::outs() << "run in: " << getPassName() << "\n";
  auto module = getOperation();
  llvm::outs() << "root op: " << module->getName() << "\n";

  RewritePatternSet buffer_cast_patterns(&getContext());
  ::mlir::north_star::populateBufferCastOpCanonicalizationPatterns(
      buffer_cast_patterns);
  GreedyRewriteConfig buffer_cast_config;
  buffer_cast_config.maxIterations = 10;
  buffer_cast_config.useTopDownTraversal = true;
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(),
          FrozenRewritePatternSet(std::move(buffer_cast_patterns)),
          buffer_cast_config)))
    signalPassFailure();

  RewritePatternSet patterns(&getContext());
  ::mlir::north_star::populateDeviceRegionFusionPatterns(patterns);
  GreedyRewriteConfig config;
  bool changed;
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), FrozenRewritePatternSet(std::move(patterns)), config,
          &changed)))
    signalPassFailure();
  llvm::outs() << "region has changed: " << changed << "\n";
  llvm::outs() << "run out: " << getPassName() << "\n\n";
}
