//    Copyright 2025 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distribute under the License is distribute on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#ifndef INTERFACES_BUILTIN_FUNCTION_INTERFACES_TD
#define INTERFACES_BUILTIN_FUNCTION_INTERFACES_TD
include "mlir/IR/Interfaces.td"

def BuiltinFunctionOpInterface: OpInterface<"BuiltinFunctionOpInterface",
                                        []>{
  let description = "内建函数的通用定义";
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/        "获取内建函数的名称",
      /*returnType=*/  "::std::string",
      /*methodName=*/  "getBuildinFunctionName",
      /*args=*/        (ins),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      return {};
      }]
      >];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

#endif // INTERFACES_DISTRIBUTE_PARALLELISM_INTERFACES_TD