set(LLVM_TARGET_DEFINITIONS NorthStarDialect.td)
# 生成NorthStar Dialect 的声明
mlir_tablegen(NorthStarDialect.h.inc --gen-dialect-decls -dialect=north_star)
# 生成NorthStar Dialect 的实现
mlir_tablegen(NorthStarDialect.cpp.inc --gen-dialect-defs -dialect=north_star)

# 将生成的命令们定义为为target
add_public_tablegen_target(NorthStarDialectIncGen${ch_num})


