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
#include "Dialect/NorthStar/NorthStarAttrs.h"

#include "Dialect/NorthStar/NorthStarDialect.h"
#include "Dialect/NorthStar/NorthStarEunms.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#define FIX
#define GET_ATTRDEF_CLASSES
#include "Dialect/NorthStar/NorthStarAttrs.cpp.inc"
#include "Dialect/NorthStar/NorthStarEunms.cpp.inc"

namespace mlir::north_star {

void NorthStarDialect::registerAttrs() {
  llvm::outs() << "register " << getDialectNamespace() << "  Attr\n";
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/NorthStar/NorthStarAttrs.cpp.inc"
      >();
}

bool LayoutAttr::isChannelLast() { return getValue() == Layout::NHWC; }
}  // namespace mlir::north_star

#undef FIX
