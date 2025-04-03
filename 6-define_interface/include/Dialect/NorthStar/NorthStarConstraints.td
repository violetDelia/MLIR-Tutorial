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
#ifndef DIALECT_NORTH_STAR_CONSTRAINTS_TD
#define DIALECT_NORTH_STAR_CONSTRAINTS_TD
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def AnyNSTensor : Type<And<[CPred<"::mlir::isa<::mlir::north_star::NSTensorType>($_self)">,
                            CPred<"::mlir::cast<::mlir::north_star::NSTensorType>($_self).getShape().size() > 0">]>>;

def AnyBuffer  : Type<And<[CPred<"::mlir::isa<::mlir::north_star::BufferType>($_self)">,
                            CPred<"::mlir::cast<::mlir::north_star::BufferType>($_self).getDevices().size() > 0">]>>;

#endif // DIALECT_NORTH_STAR_CONSTRAINTS_TD 