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

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
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

#define DEBUG_TYPE "apply-distribute-transform"

namespace mlir::north_star {
#define GEN_PASS_DEF_APPLYDISTRIBUTETRANSFORMPASS
#include "Dialect/NorthStar/Transforms/Passes.h.inc"

}  // namespace mlir::north_star
using namespace ::mlir;
using namespace ::mlir::north_star;

struct ApplyDistributeTransformPass
    : ::mlir::north_star::impl::ApplyDistributeTransformPassBase<
          ApplyDistributeTransformPass> {
  using ApplyDistributeTransformPassBase<
      ApplyDistributeTransformPass>::ApplyDistributeTransformPassBase;
  void runOnOperation() override;
};
void ApplyDistributeTransformPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto func = getOperation();
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("root op: {0}\n", func->getName()));
  auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
      func->getAttr(KDPAttrName));
  if (!dp_attr) llvm_unreachable("error!");
  func->walk([&](mlir::Operation* op) {
    if (auto dis_op = llvm::dyn_cast_or_null<mlir::DistributeParallelOp>(op)) {
      if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "Apply DataParallelism to {0}\n", op->getName()));
        op->erase();
      };
    }
  });
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}

std::unique_ptr<::mlir::Pass>
mlir::north_star::createApplyDistributeTransformPass() {
  return std::make_unique<ApplyDistributeTransformPass>();
}