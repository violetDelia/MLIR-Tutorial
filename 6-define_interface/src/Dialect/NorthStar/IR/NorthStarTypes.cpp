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
#include "Dialect/NorthStar/IR/NorthStarTypes.h"

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#define FIX
#define GET_TYPEDEF_CLASSES
#include "Dialect/NorthStar/IR/NorthStarTypes.cpp.inc"

namespace mlir::north_star {

void NorthStarDialect::registerTypes() {
  llvm::outs() << "register " << getDialectNamespace() << "  Type\n";
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/NorthStar/IR/NorthStarTypes.cpp.inc"
      >();
}

::llvm::LogicalResult NSTensorType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id) {
  if (device_id < 0) {
    return emitError() << " Invalid device id";
  }
  if (!elementType.isIntOrFloat()) {
    return emitError() << " Invalid element type ";
  }
  return llvm::success();
}

Type NSTensorType::parse(AsmParser &parser) {
  if (parser.parseLess()) return Type();

  SmallVector<int64_t, 4> dimensions;
  if (parser.parseDimensionList(dimensions, /*allowDynamic=*/true,
                                /*withTrailingX=*/true))
    return Type();
  // Parse the element type.
  auto typeLoc = parser.getCurrentLocation();
  Type elementType;
  if (parser.parseType(elementType)) return Type();
  // Check that array is formed from allowed types.
  if (parser.parseComma()) return Type();
  int device_id = 0;
  if (parser.parseInteger(device_id))
    if (parser.parseGreater()) return Type();
  return parser.getChecked<NSTensorType>(parser.getContext(), dimensions,
                                         elementType, device_id);
}

void NSTensorType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape()) {
    if (dim < 0) {
      printer << "?" << 'x';
    } else {
      printer << dim << 'x';
    }
  }
  printer.printType(getElementType());
  printer << ",";
  printer << getDeviceId();
  printer << ">";
}
::llvm::LogicalResult BufferType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> devices) {
  if (std::set(devices.begin(), devices.end()).size() != devices.size())
    return emitError() << "Duplicate device ids";
  for (auto id : devices) {
    if (id < 0) {
      return emitError() << "Invalid device id";
    }
  }
  return llvm::success();
}
}  // namespace mlir::north_star

#undef FIX
