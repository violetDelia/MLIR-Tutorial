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

#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Conversion/Passes.h"
#include "Pipelines/Pipelines.h"
int main(int argc, char **argv) {
  //mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  //registerAllDialects(registry);
  registry.insert<mlir::north_star::NorthStarDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::BuiltinDialect>();
  //registerAllExtensions(registry);
  mlir::north_star::registerNorthStarOptPasses();
  mlir::north_star::registerNorthStarConversionPasses();
  mlir::pipeline::registerNorthStarBasicPipelines();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NS modular optimizer driver\n", registry));
}
