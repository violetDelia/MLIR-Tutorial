//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "Conversion/Passes.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Pipelines/Pipelines.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  // mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  // registerAllDialects(registry);
  registry.insert<mlir::north_star::NorthStarDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::transform::TransformDialect>();
  mlir::pipeline::registerNorthStarBasicPipelinesExtennsion(registry);
  // registerAllExtensions(registry);
  mlir::north_star::registerNorthStarOptPasses();
  mlir::north_star::registerNorthStarConversionPasses();
  mlir::pipeline::registerNorthStarBasicPipelines();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NS modular optimizer driver\n", registry));
}
