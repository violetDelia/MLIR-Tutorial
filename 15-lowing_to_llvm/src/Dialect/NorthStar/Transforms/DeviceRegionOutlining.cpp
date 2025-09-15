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
#include <memory>

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#define DEBUG_TYPE "device-region-outlining"
namespace mlir::north_star {
#define GEN_PASS_DEF_DEVICEREGIONOUTLININGPASS
#include "Dialect/NorthStar/Transforms/Passes.h.inc"

}  // namespace mlir::north_star
using namespace ::mlir;
using namespace ::mlir::north_star;

struct DeviceRegionOutliningPass
    : ::mlir::north_star::impl::DeviceRegionOutliningPassBase<
          DeviceRegionOutliningPass> {
  using DeviceRegionOutliningPassBase<
      DeviceRegionOutliningPass>::DeviceRegionOutliningPassBase;
  void runOnOperation() override;
};

void DeviceRegionOutliningPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto module = getOperation();
  LLVM_DEBUG(
      llvm::dbgs() << llvm::formatv("root op: {0}\n", module->getName()));
}
