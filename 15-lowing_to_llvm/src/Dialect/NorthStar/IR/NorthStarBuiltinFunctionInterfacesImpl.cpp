
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

#include <string>

#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"
namespace mlir::north_star {

std::string TensorToNSTensorOp::getBuildinFunctionName() {
  static const constexpr char* name_base = "__NS__MemrefToNSMemref";
  auto input_type = getInput().getType();
  auto element_type = input_type.getElementType();
  std::string s;
  llvm::raw_string_ostream oss(s);
  oss << "_";
  element_type.print(oss);
  return name_base + oss.str();
}

std::string NSTensorToTensorOp::getBuildinFunctionName() {
  static const constexpr char* name_base = "__NS__NSMemrefToMemref";
  auto input_type = llvm::cast<ShapedType>(getInput().getType());
  auto element_type = input_type.getElementType();
  std::string s;
  llvm::raw_string_ostream oss(s);
  oss << "_";
  element_type.print(oss);
  return name_base + oss.str();
}

std::string BufferOp::getBuildinFunctionName() {
  static const constexpr char* name_base = "__NS__MakeBuffer";
  auto input_type = llvm::cast<ShapedType>(getTensors()[0].getType());
  auto element_type = input_type.getElementType();
  std::string s;
  llvm::raw_string_ostream oss(s);
  oss << "_";
  element_type.print(oss);
  return name_base + oss.str();
}

std::string GetTensorOp::getBuildinFunctionName() {
  static const constexpr char* name_base = "__NS__GetTensor";
  auto output_type = llvm::cast<ShapedType>(getResult().getType());
  auto element_type = output_type.getElementType();
  std::string s;
  llvm::raw_string_ostream oss(s);
  oss << "_";
  element_type.print(oss);
  return name_base + oss.str();
}

std::string GatherOp::getBuildinFunctionName() {
  static const constexpr char* name_base = "__NS__Gather";
  return name_base;
}

std::string ScatterOp::getBuildinFunctionName() {
  static const constexpr char* name_base = "__NS__Scatter";
  return name_base;
}
}  // namespace mlir::north_star