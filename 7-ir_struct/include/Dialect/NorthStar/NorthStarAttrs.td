//    Copyright 2024 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#ifndef DIALECT_NORTH_STAR_ATTRS_TD
#define DIALECT_NORTH_STAR_ATTRS_TD

include "mlir/IR/EnumAttr.td"
include "Dialect/NorthStar/NorthStarEunms.td"
include "Interfaces/DistributeParallelismInterfaces.td"

class NorthStar_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<NorthStar_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = dialect.name # "." # attrMnemonic;
  let genStorageClass = 1;
  let hasStorageCustomConstructor = 0;
  let skipDefaultBuilders = 0;
  let genVerifyDecl = 0;
}

def NorthStar_DataParallelism: NorthStar_Attr<"DataParallelism", "DP", [DataParallelAttr]>{
  let parameters = (ins "int64_t":$DP_nums, ArrayRefParameter<"int64_t">:$devices);
  let builders = [
    AttrBuilder<(ins "int64_t":$DP_nums),
    [{
      llvm::SmallVector<int64_t> device_ids;
      for (auto i : llvm::index_range(0, DP_nums)) {
        device_ids.push_back(i);
      }
      return $_get($_ctxt, DP_nums, device_ids);
    }]
    >
  ];
  let assemblyFormat = [{
    `<`
      `DP` `=` $DP_nums `:` $devices
    `>`
  }];
}

#endif //DIALECT_NORTH_STAR_ATTRS_TD