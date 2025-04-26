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
#ifndef DIALECT_NORTH_STAR_IR_NORTH_STAR_EUNMS_TD
#define DIALECT_NORTH_STAR_IR_NORTH_STAR_EUNMS_TD

include "mlir/IR/EnumAttr.td"
include "Dialect/NorthStar/IR/NorthStarTypes.td"


def NorthStar_LAYOUT_NCHW         : I32EnumAttrCase<"NCHW", 0>; 
def NorthStar_LAYOUT_NHWC         : I32EnumAttrCase<"NHWC", 1>;//C last
def NorthStar_Layout : I32EnumAttr<"Layout",
    "Layout of tensor",
    [NorthStar_LAYOUT_NCHW, NorthStar_LAYOUT_NHWC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::north_star";
}

def LLH_LayoutAttr : EnumAttr<NorthStar_Dialect, NorthStar_Layout, "Layout">{
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
  bool isChannelLast();
  }];
}

#endif //DIALECT_NORTH_STAR_IR_NORTH_STAR_EUNMS_TD