set(LLVM_TARGET_DEFINITIONS NorthStarAttrs.td)
# 生成NorthStar Dialect 的声明
mlir_tablegen(NorthStarDialect.h.inc --gen-dialect-decls -dialect=north_star)
# 生成NorthStar Dialect 的实现
mlir_tablegen(NorthStarDialect.cpp.inc --gen-dialect-defs -dialect=north_star)
# 生成NorthStar Type 的声明
mlir_tablegen(NorthStarTypes.h.inc -gen-typedef-decls -dialect=north_star)
# 生成NorthStar Type 的实现
mlir_tablegen(NorthStarTypes.cpp.inc -gen-typedef-defs -dialect=north_star)
# 生成NorthStar Enums 的声明
mlir_tablegen(NorthStarEunms.h.inc -gen-enum-decls -dialect=north_star)
# 生成NorthStar Enums 的实现
mlir_tablegen(NorthStarEunms.cpp.inc -gen-enum-defs -dialect=north_star)
# 生成NorthStar Attr 的声明
mlir_tablegen(NorthStarAttrs.h.inc -gen-attrdef-decls -dialect=north_star)
# 生成NorthStar Attr 的实现
mlir_tablegen(NorthStarAttrs.cpp.inc -gen-attrdef-defs -dialect=north_star)
# 将生成的命令们定义为为target
add_public_tablegen_target(NorthStarDialectIncGen${ch_num})


