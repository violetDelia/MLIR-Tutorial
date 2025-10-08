
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

#ifndef RUNTIME_BUFFER_H
#define RUNTIME_BUFFER_H
#include <cstddef>
#include <cstdint>

#include "Runtime/Core.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

enum DeviceType : int64_t {
  INT64 = 0,
  INT32 = 1,
  INT16 = 2,
  INT8 = 3,
  FLOAT64 = 4,
  FLOAT32 = 5,
  FLOAT16 = 6,
  BFLOAT16 = 7
};

struct Buffer {
  int64_t device_nums;
  int64_t* device_index;
  DeviceType* device_types;
  void** memref_ptrs;
};
#endif  // RUNTIME_BUFFER_H