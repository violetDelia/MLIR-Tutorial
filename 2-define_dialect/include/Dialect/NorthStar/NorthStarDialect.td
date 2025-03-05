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
#ifndef NORTH_STAR_DIALECT
#define NORTH_STAR_DIALECT
include "mlir/IR/DialectBase.td"

def NorthStar_Dialect : Dialect{
  // 方言的名字
  let name = "north_star";

  // 方言的概述。
  let summary = "north_star dialect summary.";

  // 方言的详细描述
  string description = " north_star dialect description.";

  // A list of dialects this dialect will load on construction as dependencies.
  // These are dialects that this dialect may involve in canonicalization
  // pattern or interfaces.
  let dependentDialects = ["::mlir::tensor::TensorDialect"];

  // A list of key/value pair representing an attribute type and a name.
  // This will generate helper classes on the dialect to be able to
  // manage discardable attributes on operations in a type safe way.
  let discardableAttrs = (ins);

  // The C++ namespace that ops of this dialect should be placed into.
  //
  // By default, uses the name of the dialect as the only namespace. To avoid
  // placing in any namespace, use "". To specify nested namespaces, use "::"
  // as the delimiter, e.g., given "A::B", ops will be placed in
  // `namespace A { namespace B { <ops> } }`.
  //
  // Note that this works in conjunction with dialect C++ code. Depending on how
  // the generated files are included into the dialect, you may want to specify
  // a full namespace path or a partial one.
  let cppNamespace = "::mlir::north_star";

  // An optional code block containing extra declarations to place in the
  // dialect declaration.
  code extraClassDeclaration = [{
    static void sayHello();
  }];

  // If this dialect overrides the hook for materializing constants.
  bit hasConstantMaterializer = 0; // 14-fold_and_canonicalization

  /// If the dialect definition provides a non-default destructor.
  /// If false, a default destructor implementation will be generated.
  bit hasNonDefaultDestructor = 1;

  // If this dialect overrides the hook for verifying operation attributes.
  let hasOperationAttrVerify = 0;

  // If this dialect overrides the hook for verifying region argument
  // attributes.
  let hasRegionArgAttrVerify = 0;

  // If this dialect overrides the hook for verifying region result attributes.
  let hasRegionResultAttrVerify = 0;

  // If this dialect should use default generated attribute parser boilerplate.
  // When set, ODS will generate declarations for the attribute parsing and
  // printing hooks in the dialect and default implementations that dispatch to
  // each individual attribute directly.
  bit useDefaultAttributePrinterParser = 0;

  // If this dialect should use default generated type parser boilerplate:
  // When set, ODS will generate declarations for the type parsing and printing
  // hooks in the dialect and default implementations that dispatch to each
  // individual type directly.
  bit useDefaultTypePrinterParser = 0;

  // If this dialect overrides the hook for canonicalization patterns.
  bit hasCanonicalizer = 0;

  // If this dialect can be extended at runtime with new operations or types.
  bit isExtensible = 0;

  // Whether inherent Attributes defined in ODS will be stored as Properties.
  bit usePropertiesForAttributes = 1;
}

#endif // NORTH_STAR_DIALECT