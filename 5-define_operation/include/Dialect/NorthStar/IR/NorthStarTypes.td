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

#ifndef DIALECT_NORTH_STAR_TYPES_TD
#define DIALECT_NORTH_STAR_TYPES_TD
include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/NorthStar/IR/NorthStarDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class NorthStar_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<NorthStar_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let typeName =  dialect.name # "." # typeMnemonic;
}


def NorthStar_TensorType : NorthStar_Type<"NSTensor","ns_tensor",[]>{
  // 概述
  let summary = " the summary of north-star tensor type";

  // 方言的详细描述
  let description = "description of north-star tensor type";

  // 参数
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "int64_t":$device_id
  );

  // 是否生成StorageClass, 无特殊情况，建议设为ture
  let genStorageClass = 1;
  
  // 不建议改动
  let hasStorageCustomConstructor = 0;

  // 额外的builder 声明
  let builders = [
    TypeBuilder<(ins 
        "::mlir::ArrayRef<int64_t>":$shape,
        "::mlir::Type":$elementType),[{
      return $_get(elementType.getContext(), shape, elementType, 0);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = "`<`$shape`,`$elementType`,`$device_id`>`";

  // 跳过默认的builder函数
  let skipDefaultBuilders = 0;

  // 是否生成类型检验的函数声明
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // using TensorType::clone;
    // using ShapedType::Trait<NSTensorType>::getElementTypeBitWidth;
    // using ShapedType::Trait<NSTensorType>::getRank;
    // using ShapedType::Trait<NSTensorType>::getNumElements;
    // using ShapedType::Trait<NSTensorType>::isDynamicDim;
    // using ShapedType::Trait<NSTensorType>::hasStaticShape;
    // using ShapedType::Trait<NSTensorType>::getNumDynamicDims;
    // using ShapedType::Trait<NSTensorType>::getDimSize;
    // using ShapedType::Trait<NSTensorType>::getDynamicDimIndex;
    // NSTensorType clone(::mlir::Type elementType) {
    //   return ::llvm::cast<NSTensorType>(cloneWith(getShape(), elementType));
    // }
  }];
}


def NorthStar_BufferType : NorthStar_Type<"Buffer","buffer",[]>{
  let summary = " the summary of north-star buffer type";

  let description = "description of north-star buffer type";

  let parameters = (ins 
    ArrayRefParameter<"int64_t">:$devices
  );

  let genStorageClass = 1;
  
  let hasStorageCustomConstructor = 0;

  let assemblyFormat = "`<`$devices`>`";

  let skipDefaultBuilders = 0;

  let genVerifyDecl = 1;
}


#endif // DIALECT_NORTH_STAR_TYPES_TD
