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
#include <memory>

#include "Dialect/NorthStar/IR/NorthStarAttrs.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Utils/Key.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "mark-distribute-parallel-parameters"

namespace mlir::north_star {
#define GEN_PASS_DEF_MARKDISTRIBUTEPARALLELPARAMETERSPASS
#include "Dialect/NorthStar/Transforms/Passes.h.inc"

}  // namespace mlir::north_star
using namespace ::mlir;
using namespace ::mlir::north_star;

struct MarkDistributeParallelParametersPass
    : ::mlir::north_star::impl::MarkDistributeParallelParametersPassBase<
          MarkDistributeParallelParametersPass> {
  using MarkDistributeParallelParametersPassBase<
      MarkDistributeParallelParametersPass>::
      MarkDistributeParallelParametersPassBase;
  void runOnOperation() override;
};

void MarkDistributeParallelParametersPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto module = getOperation();
  LLVM_DEBUG(
      llvm::dbgs() << llvm::formatv("root op: {0}\n", module->getName()));
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("DPNums: {0}\n", DPNums));
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("TPNums: {0}\n", TPNums));
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("EPNums: {0}\n", EPNums));

  if (TPNums != 1) {
    llvm::errs() << "TPNums not supported currently!\n";
    signalPassFailure();
    return;
  }
  if (DPNums != 1) {
    auto dp_attr = DataParallelismAttr::get(&getContext(), DPNums);
    module->walk(
        [&dp_attr](func::FuncOp op) { op->setAttr(KDPAttrName, dp_attr); });
  }
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}
