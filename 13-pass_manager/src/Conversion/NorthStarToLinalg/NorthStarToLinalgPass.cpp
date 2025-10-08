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

#include "Conversion/NorthStarToLinalg/NorthStarToLinalg.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "convert-north-satr-to-linalg"

namespace mlir::north_star {

#define GEN_PASS_DEF_CONVERTNORTHSTARTOLINALGPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::north_star

using namespace ::mlir;
using namespace ::mlir::north_star;

struct NorthStarToLinalgPassPass
    : public mlir::north_star::impl::ConvertNorthStarToLinalgPassBase<
          NorthStarToLinalgPassPass> {
  void runOnOperation() override;
};

void configNorthStarToLinalgTarget(ConversionTarget& target) {
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<BufferCastOp>();
  target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
    for (auto type : op->getOperandTypes()) {
      if (isa<::mlir::north_star::NSTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<DeviceKernelOp>([](DeviceKernelOp op) {
    for (auto type : op.getArgs().getTypes()) {
      if (isa<::mlir::north_star::NSTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<SoftmaxOp>([](Operation* op) {
    return !llvm::isa<DeviceKernelOp>(op->getParentOp());
  });
}
void NorthStarToLinalgPassPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto module = getOperation();
  TypeConverter type_convert;
  initNorthStarToLinalgTypeConvert(type_convert);
  RewritePatternSet patterns(&getContext());
  populateNorthStarToLinalgPatterns(type_convert, patterns);
  ConversionTarget target(getContext());
  configNorthStarToLinalgTarget(target);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}