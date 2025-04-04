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
#ifndef INTERFACES_DISTRIBUTE_PARALLELISM_INTERFACES_TD
#define INTERFACES_DISTRIBUTE_PARALLELISM_INTERFACES_TD
include "mlir/IR/Interfaces.td"

def DistributeParallelAttr: AttrInterface<"DistributeParallelAttr">{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir";
  let methods = [];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

def DataParallelAttr: AttrInterface<"DataParallelAttr",[DistributeParallelAttr]>{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<[{
      DP 数量.
    }],
    "int64_t", "getDPNums">,

    InterfaceMethod<[{
      设备编号.
    }],
    "::llvm::ArrayRef<int64_t>", "getDevices">,
  ];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

def SupportedDataParallelismOp: OpInterface<"SupportedDataParallelismOp">{
  let description = "Properties related to data parallelism";
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/        "进行数据并行的变换",
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "applyDataParallelism",
      /*args=*/        (ins "::mlir::DistributeParallelAttr":$attr),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      return llvm::failure();
      }]
      >,
      InterfaceMethod<
      /*desc=*/        "进行数据并行的变换",
      /*returnType=*/  "bool",
      /*methodName=*/  "supportedDataParallelism",
      /*args=*/        (ins),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      Operation* op = $_op.getOperation();
      if (op->getNumOperands() == 0) return true;
      auto base_type = op->getOperand(0).getType();
      if (!isa<mlir::ShapedType>(base_type)) return false;
      for (auto operand : op->getOperands()) {
        if (operand.getType() != base_type) return false;
      }
      return true;
      }]
      >
      ];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

def DistributeParallelOp:OpInterface<"DistributeParallelOp">{
  let description = "Properties related to distribute parallelism";
  let cppNamespace = "::mlir";
  let methods = [];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = [{
    // 实现并行变换
    ::mlir::LogicalResult applyDistributeParallelism(const ::mlir::DistributeParallelAttr attr){
      if (isa<mlir::DataParallelAttr>(attr)) {
        if (!isa<mlir::SupportedDataParallelismOp>($_op.getOperation())) return ::llvm::failure();
        return dyn_cast<mlir::SupportedDataParallelismOp>($_op.getOperation()).applyDataParallelism(attr);
      } else {
        llvm_unreachable("unsupported parallel type!");
      }
      return ::llvm::failure();
    };

    bool supportedDistributeParallelism(){
      if (isa<mlir::SupportedDataParallelismOp>($_op.getOperation())){
        return dyn_cast<mlir::SupportedDataParallelismOp>($_op.getOperation()).supportedDataParallelism();
      }else{
        llvm_unreachable("unsupported parallel type!");
      }
      return false;
    }
  }];
}

#endif // INTERFACES_DISTRIBUTE_PARALLELISM_INTERFACES_TD