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
//

#ifndef CONVERSION_NORTHSTARCOMMON_NORTHSTARBUILDER_H
#define CONVERSION_NORTHSTARCOMMON_NORTHSTARBUILDER_H

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
namespace mlir::north_star {

class NSTensorDescriptor : public StructBuilder {
 public:
  explicit NSTensorDescriptor(Value descriptor);
  static NSTensorDescriptor undef(OpBuilder &builder, Location loc,
                                        Type descriptorType);

};

}  // namespace mlir::north_star

#endif  // CONVERSION_NORTHSTARCOMMON_NORTHSTARBUILDER_H
