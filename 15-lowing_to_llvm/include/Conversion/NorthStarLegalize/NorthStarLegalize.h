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

#ifndef CONVERSION_NORTHSTARLEGALIZE_NORTHSTARLEGALIZE_H
#define CONVERSION_NORTHSTARLEGALIZE_NORTHSTARLEGALIZE_H
#include <memory>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
namespace mlir {
class TypeConverter;
}
namespace mlir::north_star {

void initNorthStarLegalizeTypeConvert(TypeConverter &type_converter);

void populateNorthStarLegalizePatterns(TypeConverter &type_converter,
                                     RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTNORTHSTARLEGALIZEPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::north_star
#endif  // CONVERSION_NORTHSTARLEGALIZE_NORTHSTARLEGALIZE_H