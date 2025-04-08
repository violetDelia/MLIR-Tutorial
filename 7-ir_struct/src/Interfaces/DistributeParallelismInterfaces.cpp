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
#include "Interfaces/DistributeParallelismInterfaces.h"

#include <cstdint>

#include "Interfaces/DistributeParallelismAttrInterfaces.cpp.inc"
#include "Interfaces/DistributeParallelismOpInterfaces.cpp.inc"
#include "Dialect/NorthStar/NorthStarTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
void test() {
  int DP_nums;
  llvm::SmallVector<int64_t> device_ids;
  for (auto i : llvm::index_range(0, DP_nums)) {
    device_ids.push_back(i);
  }
}
