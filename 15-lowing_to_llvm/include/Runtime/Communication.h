
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

#ifndef RUNTIME_COMMUNICATION_H
#define RUNTIME_COMMUNICATION_H
#include <cstddef>
#include <cstdint>

#include "Runtime/Buffer.h"
#include "Runtime/Core.h"
#include "Runtime/Tensor.h"

MLIR_TT_C_EXPORT Buffer __NS__MakeBuffer_f32(NSMemref<float>* memref1,
                                             int64_t* device_index1,
                                             int64_t device_num);

MLIR_TT_C_EXPORT NSMemref<float> __NS__GetTensor_f32(int64_t device_id,
                                                     Buffer& buffer);

MLIR_TT_C_EXPORT void __NS__Scatter(Buffer& src_buffer, Buffer& dst_buffer,
                                    void* exetra_params);

MLIR_TT_C_EXPORT void __NS__Gather(Buffer& src_buffer, Buffer& dst_buffer,
                                    void* exetra_params);

#endif  // RUNTIME_COMMUNICATION_H