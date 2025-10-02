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

#include "Conversion/NorthStarLegalize/NorthStarLegalize.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Utils/Key.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "convert-north-satr-to-func"

namespace mlir::north_star {

#define GEN_PASS_DEF_NORTHSTARLEGALIZEPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::north_star

using namespace ::mlir;
using namespace ::mlir::north_star;

struct NorthStarLegalizePassPass
    : public mlir::north_star::impl::NorthStarLegalizePassBase<
          NorthStarLegalizePassPass> {
  void runOnOperation() override;
};

void configNorthStarLegalizeTarget(ConversionTarget& target) {
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<BufferOp, TensorToNSTensorOp, NSTensorToTensorOp,ScatterOp,GatherOp,GetTensorOp>();
  target.addIllegalOp<DeviceKernelOp, ReturnOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
    auto func_type = op.getFunctionType();
    for (auto type : func_type.getInputs()) {
      if (isa<::mlir::north_star::NSTensorType>(type)) return false;
    }
    for (auto type : func_type.getResults()) {
      if (isa<::mlir::north_star::NSTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
    for (auto type : op->getOperandTypes()) {
      if (isa<::mlir::north_star::NSTensorType>(type)) return false;
    }
    return true;
  });
}

void NorthStarLegalizePassPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
  auto model = getOperation();
  auto main_func = model.lookupSymbol<func::FuncOp>(KEntryPointName);
  if (!main_func || !main_func.isPublic()) {
    model.emitError() << "Cannot find host entry function";
    signalPassFailure();
    return;
  }
  TypeConverter type_convert;
  initNorthStarLegalizeTypeConvert(type_convert);
  RewritePatternSet patterns(&getContext());
  populateNorthStarLegalizePatterns(type_convert, patterns);
  ConversionTarget target(getContext());
  configNorthStarLegalizeTarget(target);
  if (failed(applyPartialConversion(model, target, std::move(patterns))))
    signalPassFailure();
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}