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

#include "Dialect/NorthStar/IR/NorthStarOps.h"

#include <algorithm>

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Value.h"
#define GET_OP_CLASSES
#include "Dialect/NorthStar/IR/NorthStarOps.cpp.inc"

namespace mlir::north_star {

void NorthStarDialect::registerOps() {
  llvm::outs() << "register " << getDialectNamespace() << "  Op\n";
  addOperations<
#define GET_OP_LIST
#include "Dialect/NorthStar/IR/NorthStarOps.cpp.inc"
      >();
}

::llvm::LogicalResult GetTensorOp::verify() {
  auto device_id = getDeviceId();
  auto buffer = getBuffer();
  if (isa<BlockArgument>(buffer)) {
    auto buffer_type = cast<BufferType>(buffer.getType());
    auto device_ids = buffer_type.getDevices();
    for (auto id : device_ids) {
      if (id == device_id) return llvm::success();
    }
    return llvm::failure();
  }
  auto buffer_op = llvm::cast_or_null<BufferOp>(buffer.getDefiningOp());
  if (!buffer_op) return llvm::failure();
  for (auto tensor : buffer_op.getTensors()) {
    auto tensor_type = cast_or_null<NSTensorType>(tensor.getType());
    if (!tensor_type) return llvm::failure();
    if (device_id == tensor_type.getDeviceId()) {
      if (tensor_type != getType()) return llvm::failure();
      return llvm::success();
    }
  }
  return llvm::failure();
};

::llvm::LogicalResult BufferOp::verify() {
  auto tensors = getTensors();
  auto devices = cast<BufferType>(getType()).getDevices();
  if (tensors.size() == 0) return llvm::failure();
  for (auto [index, device_id, tensor] : llvm::enumerate(devices, tensors)) {
    auto tensor_type = cast_or_null<NSTensorType>(tensor.getType());
    if (device_id != tensor_type.getDeviceId()) return llvm::failure();
  }
  return llvm::success();
}

::llvm::LogicalResult SoftmaxOp::verify() {
  auto axis = getAxis();
  if (axis < 0) return llvm::failure();
  auto input_type = cast<NSTensorType>(getInput().getType());
  if (axis >= input_type.getShape().size()) return llvm::failure();
  return llvm::success();
}

}  // namespace mlir::north_star