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

#ifndef UTILS_MLIR_UTILS_KEY_H
#define UTILS_MLIR_UTILS_KEY_H

inline static const char* KEntryPointName = "main";
inline static const char* KDPAttrName = "dp_attr";
inline static const char* KHostFunc = "host_func";
inline static const char* KDeviceFunc = "device_kernel";
inline static const char* KDeviceIdAttr = "device_id";
inline static const char* KFuncDeviceIdAttr = "func.device_id";
inline static const char* KDevcieKernelSuffix = "_device_";

//
inline static const char* KMemrefCopyName = "memrefCopy";
inline static const char* KFreeName = "free";
inline static const char* KMallocName = "malloc";
//
inline static const char* KSetDeviceBuiltinName = "__NS__SetDevice";
inline static const char* KMemrefCopyBuiltinName = "__NS__MemrefCopy";
inline static const char* KFreeBuiltinName = "__NS__Free";
inline static const char* KMallocBuiltinName= "__NS__Malloc";
inline static const char* KMemcpyBuiltinName= "__NS__Memcpy";
#endif  // UTILS_MLIR_UTILS_KEY_H
