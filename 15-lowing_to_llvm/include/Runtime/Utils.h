
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

#ifndef RUNTIME_UTILS_H
#define RUNTIME_UTILS_H

#include <cstdint>

#include "Runtime/Core.h"
#include "Runtime/Tensor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

MLIR_TT_C_EXPORT NSMemref<float> __NS__MemrefToNSMemref_f32(
    int64_t device_id, UnrankedMemRefType<float> *memref);

MLIR_TT_C_EXPORT UnrankedMemRefType<float> __NS__NSMemrefToMemref_f32(
    NSMemref<float> &memref);
#endif  // RUNTIME_UTILS_H