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

#include "Dialect/NorthStar/NorthStarOps.h"

#include <algorithm>

#include "Dialect/NorthStar/NorthStarDialect.h"
#include "Dialect/NorthStar/NorthStarTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Value.h"
#define GET_OP_CLASSES
#include "Dialect/NorthStar/NorthStarOps.cpp.inc"
namespace mlir::north_star {
void NorthStarDialect::registerOps() {
  llvm::outs() << "register " << getDialectNamespace() << "  Ops\n";
  addOperations<
#define GET_OP_LIST
#include "Dialect/NorthStar/NorthStarOps.cpp.inc"
      >();
}

::llvm::LogicalResult GetTensorOp::verify() {
  int64_t device_id = getDeviceId();
  if (isa<BlockArgument>(getBuffer())) {
    auto buffer_type = llvm::cast_or_null<BufferType>(getBuffer().getType());
    auto device_ids = buffer_type.getDevices();
    for (auto device : device_ids) {
      if (device_id == device) return llvm::success();
    }
    return llvm::failure();
  }

  auto buffer_op = llvm::cast_or_null<BufferOp>(getBuffer().getDefiningOp());
  if (!buffer_op) return llvm::failure();
  for (auto tensor : buffer_op.getTensors()) {
    auto tensor_type = llvm::cast_or_null<NSTensorType>(tensor.getType());
    if (!tensor_type) return llvm::failure();
    if (device_id == tensor_type.getDeviceId()) {
      if (tensor_type != getType()) return llvm::failure();
      return llvm::success();
    }
  }
  return llvm::failure();
}

::llvm::LogicalResult BufferOp::verify() {
  auto inputs = getTensors();
  auto devices = cast<BufferType>(getType()).getDevices();
  if (inputs.empty()) return llvm::failure();
  for (auto [index, device_id, input] : llvm::enumerate(devices, inputs)) {
    auto tensor_type = llvm::cast_or_null<NSTensorType>(input.getType());
    if (!tensor_type) return llvm::failure();
    if (device_id != tensor_type.getDeviceId()) return llvm::failure();
  }
  return llvm::success();
}

::llvm::LogicalResult SoftmaxOp::verify() {
  auto axis = getAxis();
  auto tensor_type = llvm::cast_or_null<NSTensorType>(getType());
  if (!tensor_type) return llvm::failure();
  if (axis < 0 || axis >= tensor_type.getShape().size()) return llvm::failure();
  return llvm::success();
}

}  // namespace mlir::north_star