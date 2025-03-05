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
#include "Dialect/NorthStar/NorthStarDialect.h"

#include <iostream>

#include "llvm/Support/Printable.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#define FIX
#include "Dialect/NorthStar/NorthStarDialect.cpp.inc"
#undef FIX

namespace mlir {
namespace north_star {
void NorthStarDialect::initialize() {
  llvm::outs() << "initialize NorthStar Dialect"
               << "\n";
}

NorthStarDialect::~NorthStarDialect() {
  llvm::outs() << "deconstruct NorthStar Dialect"
               << "\n";
}

void NorthStarDialect::sayHello() {
  llvm::outs() << "hello NorthStar Dialect"
               << "\n";
}
}  // namespace north_star
}  // namespace mlir
