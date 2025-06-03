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
#ifndef INTERFACES_FUSION_REGION_INTERFACES_TD
#define INTERFACES_FUSION_REGION_INTERFACES_TD
include "mlir/IR/Interfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def FusionRegionOpInterface: OpInterface<"FusionRegionOpInterfaces",
                                        []>{
  let description = "将多个Op合并到一个Region上的泛化接口";
  let cppNamespace = "::mlir";
  let methods = [
    StaticInterfaceMethod<
      /*desc=*/        "进行数据并行的变换",
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "FusionOps",
      /*args=*/        (ins "::mlir::RewriterBase&":$rewriter,"mlir::ArrayRef<::mlir::Operation*>":$ops, "::mlir::Location":$loc),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      return llvm::failure();
      }]
      >];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

#endif // INTERFACES_FUSION_REGION_INTERFACES_TD