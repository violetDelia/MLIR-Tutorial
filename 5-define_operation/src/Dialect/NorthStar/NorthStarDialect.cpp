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

#include "llvm/Support/raw_ostream.h"
#define FIX
#include "Dialect/NorthStar/NorthStarDialect.cpp.inc"
#undef FIX

namespace mlir::north_star {
// 实现方言的初始化方法
void NorthStarDialect::initialize() {
  llvm::outs() << "initializing " << getDialectNamespace() << "\n";
  registerTypes();
  registerAttrs();
  registerOps();
}

// 实现方言的析构函数
NorthStarDialect::~NorthStarDialect() {
  llvm::outs() << "destroying " << getDialectNamespace() << "\n";
}

// 实现在extraClassDeclaration 声明当中生命的方法。
void NorthStarDialect::sayHello() {
  llvm::outs() << "Hello in " << getDialectNamespace() << "\n";
}

}  // namespace mlir::north_star
