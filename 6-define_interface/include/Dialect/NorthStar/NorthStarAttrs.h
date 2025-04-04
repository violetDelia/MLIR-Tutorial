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

#ifndef DIALECT_NORTH_STAR_ATTRS_H
#define DIALECT_NORTH_STAR_ATTRS_H
#include "Dialect/NorthStar/NorthStarEunms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "Dialect/Interfaces/DistributeParallelismInterfaces.h"
#define GET_ATTRDEF_CLASSES
#include "Dialect/NorthStar/NorthStarAttrs.h.inc"

#endif  // DIALECT_NORTH_STAR_ATTRS_H