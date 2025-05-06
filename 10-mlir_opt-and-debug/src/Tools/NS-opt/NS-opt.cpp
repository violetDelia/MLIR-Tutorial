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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir-c/Debug.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::north_star::NorthStarDialect>();
  registerAllExtensions(registry);
  mlir::north_star::registerNorthStarOptPasses();
  // mlirEnableGlobalDebug(true);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NS modular optimizer driver\n", registry));
}
