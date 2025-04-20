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
#include <cstddef>

#include "Dialect/NorthStar/IR/NorthStarAttrs.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

void CH2() {
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  // 调用方言中的方法
  dialect->sayHello();
}

void typeBrief() {
  // 文件定义：llvm-project/mlir/include/mlir/IR/BuiltinTypes.td
  auto context = new mlir::MLIRContext;

  // 浮点数，每种位宽和标准定义一个
  auto f32 = mlir::Float32Type::get(context);
  llvm::outs() << "F32类型 :\t";
  f32.dump();

  auto bf16 = mlir::BFloat16Type::get(context);
  llvm::outs() << "BF16类型 :\t";
  bf16.dump();

  // Index 类型，机器相关的整数类型
  auto index = mlir::IndexType::get(context);
  llvm::outs() << "Index 类型 :\t";
  index.dump();

  // 整数类型, 参数: 位宽&&有无符号
  auto i32 = mlir::IntegerType::get(context, 32);
  llvm::outs() << "I32 类型 :\t";
  i32.dump();
  auto ui16 = mlir::IntegerType::get(context, 16, mlir::IntegerType::Unsigned);
  llvm::outs() << "UI16 类型 :\t";
  ui16.dump();

  // 张量类型,表示的是数据，不会有内存的布局信息。
  auto static_tensor = mlir::RankedTensorType::get({1, 2, 3}, f32);
  llvm::outs() << "静态F32 张量类型 :\t";
  static_tensor.dump();
  // 动态张量
  auto dynamic_tensor =
      mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 2, 3}, f32);
  llvm::outs() << "动态F32 张量类型 :\t";
  dynamic_tensor.dump();

  // Memref类型：表示内存
  auto basic_memref = mlir::MemRefType::get({1, 2, 3}, f32);
  llvm::outs() << "静态F32 内存类型 :\t";
  basic_memref.dump();
  // 带有布局信息的内存

  auto stride_layout_memref = mlir::MemRefType::get(
      {1, 2, 3}, f32, mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}));
  llvm::outs() << "连续附带布局信息的 F32 内存类型 :\t";
  stride_layout_memref.dump();
  // 使用affine 表示布局信息的内存
  auto affine_memref = mlir::MemRefType::get(
      {1, 2, 3}, f32,
      mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}).getAffineMap());
  llvm::outs() << "连续附带 affine 布局信息的 F32 内存类型 :\t";
  affine_memref.dump();
  // 动态连续附带 affine 布局信息的内存
  auto dynamic_affine_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap());
  llvm::outs() << "连续附带 affine 布局信息的动态 F32 内存类型 :\t";
  dynamic_affine_memref.dump();
  // 具有内存层级信息的内存
  auto L1_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap(),
                            1);
  llvm::outs() << "处于L1层级的 F32 内存类型 :\t";
  L1_memref.dump();
  // gpu 私有内存层级的内存
  context->getOrLoadDialect<mlir::gpu::GPUDialect>();
  auto gpu_memref =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic, 2, 3}, f32,
                            mlir::StridedLayoutAttr::get(
                                context, 1, {mlir::ShapedType::kDynamic, 3, 1})
                                .getAffineMap(),
                            mlir::gpu::AddressSpaceAttr::get(
                                context, mlir::gpu::AddressSpace::Private));
  llvm::outs() << "连续附带 affine 布局信息的动态 F32 Gpu Private内存类型 :\t";
  gpu_memref.dump();

  // 向量类型,定长的一段内存
  auto vector_type = mlir::VectorType::get(3, f32);
  llvm::outs() << "F32 1D向量类型 :\t";
  vector_type.dump();

  auto vector_2D_type = mlir::VectorType::get({3, 3}, f32);
  llvm::outs() << "F32 2D向量类型 :\t";
  vector_2D_type.dump();
  delete context;
}

void CH3() {
  typeBrief();
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  // 静态 NSTensor
  mlir::north_star::NSTensorType ns_tensor =
      mlir::north_star::NSTensorType::get(&context, {1, 2, 3},
                                          mlir::Float32Type::get(&context), 3);
  llvm::outs() << "North Star Tensor 类型 :\t";
  ns_tensor.dump();
  // 动态 NSTensor
  mlir::north_star::NSTensorType dy_ns_tensor =
      mlir::north_star::NSTensorType::get(&context,
                                          {mlir::ShapedType::kDynamic, 2, 3},
                                          mlir::Float32Type::get(&context), 3);
  llvm::outs() << "动态 North Star Tensor 类型 :\t";
  dy_ns_tensor.dump();
}

void attributeBrief() {
  auto context = new mlir::MLIRContext;
  context->getOrLoadDialect<mlir::north_star::NorthStarDialect>();

  // Float Attr  表示浮点数的Attribute
  auto f32_attr = mlir::FloatAttr::get(mlir::Float32Type::get(context), 2);
  llvm::outs() << "F32 Attribute :\t";
  f32_attr.dump();

  // Integer Attr  表示整数的Attribute
  auto i32_attr =
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), 10);
  llvm::outs() << "I32 Attribute :\t";
  i32_attr.dump();

  // StrideLayout Attr  表示内存布局信息的Attribute
  auto stride_layout_attr = mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1});
  llvm::outs() << "StrideLayout Attribute :\t";
  stride_layout_attr.dump();

  // String Attr    表示字符串的Attribute
  auto str_attr = mlir::StringAttr::get(context, "Hello, MLIR!");
  llvm::outs() << "String Attribute :\t";
  str_attr.dump();

  // StrRef Attr   表示符号的Attribute
  auto str_ref_attr = mlir::SymbolRefAttr::get(str_attr);
  llvm::outs() << "SymbolRef Attribute :\t";
  str_ref_attr.dump();

  // Type Attr    储存Type 的Attribute
  auto type_attr = mlir::TypeAttr::get(mlir::north_star::NSTensorType::get(
      context, {1, 2, 3}, mlir::Float32Type::get(context)));
  llvm::outs() << "Type Attribute :\t";
  type_attr.dump();

  // Unit Attr   一般作为标记使用
  auto unit_attr = mlir::UnitAttr::get(context);
  llvm::outs() << "Unit Attribute :\t";
  unit_attr.dump();

  auto i64_arr_attr = mlir::DenseI64ArrayAttr::get(context, {1, 2, 3});
  llvm::outs() << "Array Attribute :\t";
  i64_arr_attr.dump();

  auto dense_attr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({2, 2}, mlir::Float32Type::get(context)),
      llvm::ArrayRef<float>{1, 2, 3, 4});
  llvm::outs() << "Dense Attribute :\t";
  dense_attr.dump();
  delete context;
}

void CH4() {
  attributeBrief();
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  // Layout Eunms
  auto nchw = mlir::north_star::Layout::NCHW;
  llvm::outs() << "NCHW: " << mlir::north_star::stringifyEnum(nchw) << "\n";
  // LayoutAttr
  auto nchw_attr = mlir::north_star::LayoutAttr::get(&context, nchw);
  llvm::outs() << "NCHW LayoutAttribute :\t";
  nchw_attr.dump();
  // DataParallelismAttr
  auto dp_attr = mlir::north_star::DataParallelismAttr::get(&context, 2);
  llvm::outs() << "DataParallelism Attribute :\t";
  dp_attr.dump();
}
int main() { CH4(); }