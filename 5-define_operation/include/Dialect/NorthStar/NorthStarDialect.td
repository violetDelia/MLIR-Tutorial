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

#ifndef DIALECT_NORTH_STAR
#define DIALECT_NORTH_STAR
include "mlir/IR/DialectBase.td"
include "Dialect/NorthStar/NorthStarConstraints.td"

def NorthStar_Dialect : Dialect{
  // 方言的名字
  let name = "north_star";

  // 方言的概述
  let summary = "summary of NorthStar Dialect";

  // 方言的详细描述
  let description = "description of NorthStar Dialect";

  // 方言的依赖
  let dependentDialects = ["::mlir::tensor::TensorDialect"];

  // 用于生成比较标准的属性管理的代码 [4-7]
  let discardableAttrs = (ins);

  // 生成代码的命名空间
  let cppNamespace = "::mlir::north_star";

  // 额外的声明.
  let extraClassDeclaration = [{
    static void sayHello();
    void registerTypes();
    void registerAttrs();
    void registerOps();
  }];

  // 规范化的声明. [14]
  let hasConstantMaterializer = 0;

  // 是否生成默认的析构函数
  let hasNonDefaultDestructor = 1;

  // 操作数的属性检验 [7]
  let hasOperationAttrVerify = 0;

  // RegionArg的属性检验 [7]
  let hasRegionArgAttrVerify = 0;

  // RegionResult的属性检验 [7]
  let hasRegionResultAttrVerify = 0;

  // [6]
  let hasOperationInterfaceFallback = 0;

  // 使用MLIR默认的属性解析输出.
  let useDefaultAttributePrinterParser = 1;

  // 使用MLIR默认的类型解析输出.
  let useDefaultTypePrinterParser = 1;

  // 是否有规范化patten[14].
  let hasCanonicalizer = 0;

  // 是否是可扩展的方言.
  let isExtensible = 0;

  // Whether inherent Attributes defined in ODS will be stored as Properties.
  let usePropertiesForAttributes = 1;

}

#endif // DIALECT_NORTH_STAR
