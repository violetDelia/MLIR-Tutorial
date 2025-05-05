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
#include <map>
#include <string>
#include <utility>

#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
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

::llvm::LogicalResult BufferCastOp::verify() {
  if (getNumResults() > 1) {
    if (!llvm::all_of(getResultTypes(),
                      [](Type type) { return isa<NSTensorType>(type); }))
      return llvm::failure();
  }
  if (getNumOperands() > 1) {
    if (!llvm::all_of(getOperandTypes(),
                      [](Type type) { return isa<NSTensorType>(type); }))
      return llvm::failure();
  }
  return llvm::success();
}

llvm::SmallVector<Type> splitTensor(const NSTensorType& tensor, int dim,
                                    llvm::ArrayRef<int64_t> device_ids) {
  llvm::SmallVector<Type> types;
  if (tensor.getRank() <= dim) {
    llvm::errs() << "out of dimensions rangesn";
    return {};
  }

  auto shapes = tensor.getShape();
  auto nums = device_ids.size();
  auto split_dim = shapes[dim];
  for (auto device_id : device_ids) {
    llvm::SmallVector<int64_t> new_shape(shapes.begin(), shapes.end());
    if (split_dim != ShapedType::kDynamic) {
      auto dim_value = split_dim / nums;
      new_shape[dim] = dim_value;
      split_dim -= dim_value;
      nums--;
    }
    auto new_tensor = tensor.clone(new_shape, device_id);
    types.push_back(new_tensor);
  }
  return types;
}

::llvm::LogicalResult SoftmaxOp::applyDataParallelism(
    ::mlir::DistributeParallelAttr attr) {
  auto dp_attr = llvm::dyn_cast_or_null<::mlir::DataParallelAttr>(attr);
  if (!dp_attr) return llvm::failure();
  if (!supportedDataParallelism()) return llvm::failure();
  auto op = getOperation();
  auto dp_num = dp_attr.getDPNums();
  auto device_ids = dp_attr.getDevices();
  OpBuilder builder(getOperation());
  builder.setInsertionPointAfter(getOperation());
  auto operands = getOperation()->getOperands();
  auto results = getOperation()->getResults();

  llvm::SmallVector<Operation*> ops;
  llvm::for_each(device_ids, [&](int64_t) {
    ops.push_back(builder.clone(*getOperation()));
  });
  for (auto [index, operand] : llvm::enumerate(operands)) {
    auto type = llvm::dyn_cast_or_null<NSTensorType>(operand.getType());
    auto types = splitTensor(type, 0, device_ids);
    auto cast = builder.create<north_star::BufferCastOp>(
        getLoc(), TypeRange(types), ValueRange{operand}, attr);
    cast->moveAfter(op);
    for (auto [op_index, sub_op] : llvm::enumerate(ops)) {
      sub_op->setOperand(index, cast.getResult(op_index));
    }
  }
  for (auto [index, res] : llvm::enumerate(results)) {
    auto type = llvm::dyn_cast_or_null<NSTensorType>(res.getType());
    auto types = splitTensor(type, 0, device_ids);
    for (auto [op_index, sub_op] : llvm::enumerate(ops)) {
      sub_op->getResult(index).setType(types[op_index]);
    }
    llvm::SmallVector<Value> oprands;
    for (auto sub_op : ops) {
      oprands.push_back(sub_op->getResult(index));
    }
    auto cast = builder.create<north_star::BufferCastOp>(
        getLoc(), TypeRange{type}, oprands, attr);
    for (auto& use : res.getUses()) {
      use.set(cast->getOpResult(0));
    }
  }
  return llvm::success();
}

bool SoftmaxOp::supportedDataParallelism() { return getAxis() != 0; }

::llvm::LogicalResult DeviceKernelOp::verify() { return llvm::success(); }

namespace {
static inline llvm::SmallString<4> getFusionName(
    mlir::ArrayRef<::mlir::Operation*> ops) {
  llvm::SmallString<4> name;
  for (auto op : ops) {
    name.append(op->getName().stripDialect());
    name.append("_");
    for (auto type : op->getOperandTypes()) {
      if (auto shaped = llvm::dyn_cast_or_null<ShapedType>(type)) {
        for (auto index : llvm::index_range(0, shaped.getRank())) {
          if (shaped.isDynamicDim(index)) {
            name.append("d_");
          } else {
            name.append(llvm::to_string(shaped.getDimSize(index)));
            name.append("_");
          }
        }
      }
    }
  }
  return name;
}

static inline int getDeviceid(mlir::ArrayRef<::mlir::Operation*> ops) {
  if (auto tensor = llvm::cast_or_null<north_star::NSTensorType>(
          ops.back()->getResultTypes().front())) {
    return tensor.getDeviceId();
  }
  llvm_unreachable("");
  return -1;
}

static inline llvm::MapVector<Value, std::pair<Operation*, int>>
getFusionInputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> op_set(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> res;
  for (auto op : ops) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand))
        res[operand] = std::make_pair(nullptr, 0);
      if (op_set.contains(operand.getDefiningOp())) continue;
      res[operand] = std::make_pair(op, index);
    }
  }
  return res;
}

static inline llvm::MapVector<Value, std::pair<Operation*, int>>
getFusionOutputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> op_set(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> outs;
  for (auto op : ops) {
    for (auto [index, res] : llvm::enumerate(op->getResults())) {
      for (auto user : res.getUsers()) {
        if (op_set.contains(user)) continue;
        outs[res] = std::make_pair(op, index);
        break;
      }
    }
  }
  return outs;
}
}  // namespace
llvm::LogicalResult DeviceKernelOp::FusionOps(::mlir::RewriterBase& rewriter,
                               mlir::ArrayRef<::mlir::Operation*> ops,
                               ::mlir::Location loc) {
  if (ops.size() == 0) return llvm::failure();
  auto name = getFusionName(ops);
  auto device_id = getDeviceid(ops);
  Block block;

  auto inputs_map = getFusionInputs(ops);
  auto outputs_map = getFusionOutputs(ops);
  llvm::SmallVector<Value> inputs_val;
  llvm::SmallVector<Value> output_val;
  llvm::SmallVector<Type> outputs_type;
  for (auto [key, val] : inputs_map) {
    inputs_val.push_back(key);
  }
  for (auto [key, val] : outputs_map) {
    outputs_type.push_back(key.getType());
  }
  auto kernel = rewriter.create<DeviceKernelOp>(loc, outputs_type, name,
                                                device_id, inputs_val);
  kernel->getRegion(0).push_back(&block);
  std::map<Operation*, Operation*> op_map;
  for (auto op : ops) {
    auto clone_op = op->clone();
    block.push_back(clone_op);
    op_map[op] = clone_op;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand)) continue;
      if (op_map.contains(operand.getDefiningOp())) {
        op_map[op]->setOperand(
            index,
            op_map[operand.getDefiningOp()]->getResult(
                llvm::cast_or_null<OpResult>(operand).getResultNumber()));
      }
    }
  }
  for (auto [key, val] : outputs_map) {
    output_val.push_back(op_map[val.first]->getResult(val.second));
  }
  for (auto [index, key] : llvm::enumerate(inputs_map)) {
    auto arg = block.addArgument(key.first.getType(), loc);
    op_map[key.second.first]->setOperand(key.second.second, arg);
  }
  auto insert_point = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(&block);
  rewriter.create<ReturnOp>(loc, output_val);
  rewriter.setInsertionPoint(insert_point.getBlock(), insert_point.getPoint());
  for (auto [index, key] : llvm::enumerate(outputs_map)) {
    rewriter.replaceAllUsesWith(key.first, kernel->getResult(index));
  }
  kernel->getParentOp()->dump();;
  return llvm::success();
}
}  // namespace mlir::north_star
