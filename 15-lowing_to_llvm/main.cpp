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
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "Dialect/NorthStar/IR/NorthStarAttrs.h"
#include "Dialect/NorthStar/IR/NorthStarDialect.h"
#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Dialect/NorthStar/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/File.h"
#include "Utils/Key.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

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

void CH5() {
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // ModuleOp
  auto module = builder.create<mlir::ModuleOp>(loc, "NorthStar");
  builder.setInsertionPointToStart(module.getBody());
  // ConstOp
  auto f32 = mlir::Float32Type::get(&context);
  auto shape = mlir::SmallVector<int64_t>({2, 2});
  auto const_value_1 =
      mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)1));
  auto const_value_2 =
      mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)2));
  auto tensor_type_1 =
      mlir::north_star::NSTensorType::get(&context, shape, f32, 0);
  auto tensor_type_2 =
      mlir::north_star::NSTensorType::get(&context, shape, f32, 1);
  auto const_1 = builder.create<mlir::north_star::ConstOp>(
      loc, tensor_type_1,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_1));
  auto const_2 = builder.create<mlir::north_star::ConstOp>(
      loc, tensor_type_1,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_1));
  auto const_3 = builder.create<mlir::north_star::ConstOp>(
      loc, tensor_type_2,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_2));
  auto const_4 = builder.create<mlir::north_star::ConstOp>(
      loc, tensor_type_2,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_2));
  llvm::outs() << "Const tensor in divece 0 :\n";
  const_1->dump();
  llvm::outs() << "Const tensor in divece 1 :\n";
  const_3->dump();
  // Buffer Op
  auto buffer_op = builder.create<mlir::north_star::BufferOp>(
      loc, mlir::ValueRange({const_1, const_3}));
  llvm::outs() << "Buffer Op :\n";
  buffer_op->dump();
  // Get Tensor Op
  auto get_tensor_op_1 = builder.create<mlir::north_star::GetTensorOp>(
      loc, tensor_type_1, buffer_op, 0);
  auto get_tensor_op_2 = builder.create<mlir::north_star::GetTensorOp>(
      loc, tensor_type_2, buffer_op, 1);
  llvm::outs() << "Get Tensor Op :\n";
  get_tensor_op_1->dump();
  get_tensor_op_2->dump();
  // Softmax Op
  auto softmax_op =
      builder.create<mlir::north_star::SoftmaxOp>(loc, get_tensor_op_1, 1);
  llvm::outs() << "Softmax Op :\n";
  softmax_op->dump();
  // Exp Op
  auto exp_op = builder.create<mlir::north_star::ExpOp>(loc, get_tensor_op_2);
  llvm::outs() << "Exp Op :\n";
  exp_op->dump();
  // all to all op
  auto out_buffer_op = builder.create<mlir::north_star::BufferOp>(
      loc, mlir::ValueRange({const_2, const_4}));
  auto all_to_all_op = builder.create<mlir::north_star::AllToAllOp>(
      loc, buffer_op, out_buffer_op);
  llvm::outs() << "All to All Op :\n";
  all_to_all_op->dump();
}

mlir::ModuleOp getModule(mlir::OpBuilder& builder) {
  auto loc = builder.getUnknownLoc();
  auto context = builder.getContext();
  auto module = builder.create<mlir::ModuleOp>(loc, "NorthStar");
  builder.setInsertionPointToStart(module.getBody());
  auto f32 = mlir::Float32Type::get(context);
  auto dy_dim = 128;
  auto dy_shape = mlir::SmallVector<int64_t>({dy_dim, dy_dim, 24});
  auto dy_tensor_type =
      mlir::north_star::NSTensorType::get(context, dy_shape, f32, 0);
  auto func_type =
      mlir::FunctionType::get(context, {dy_tensor_type}, {dy_tensor_type});
  auto func =
      builder.create<mlir::func::FuncOp>(loc, KEntryPointName, func_type);
  func->setAttr(KHostFunc, builder.getUnitAttr());
  func->setAttr(KDPAttrName,
                mlir::north_star::DataParallelismAttr::get(context, 2));

  auto block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  // Softmax Op
  mlir::Value softmax_op = builder.create<mlir::north_star::SoftmaxOp>(
      loc, block->getArgument(0), 1);
  softmax_op = builder.create<mlir::north_star::SoftmaxOp>(loc, softmax_op, 1);
  builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{softmax_op});
  return module;
}
void CH6() {
  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  // shaped type interface
  auto f32 = mlir::Float32Type::get(&context);
  auto dim = mlir::ShapedType::kDynamic;
  auto shape = mlir::SmallVector<int64_t>({dim, dim, 24});
  auto tensor_type =
      mlir::north_star::NSTensorType::get(&context, shape, f32, 0);
  auto shaped_type = mlir::cast<mlir::ShapedType>(tensor_type);
  llvm::outs() << "NSTensorType: \t";
  tensor_type.dump();
  llvm::outs() << "Shaped Type Interface:\t";
  shaped_type.dump();
  auto cloned_type = shaped_type.clone(f32);
  llvm::outs() << "Cloned Shaped Type Interface:\t";
  cloned_type.dump();
  // Attr interface
  auto dp_attr = mlir::north_star::DataParallelismAttr::get(&context, 2);
  llvm::outs() << dp_attr.getAbstractAttribute().getName()
               << " has mlir::DataParallelAttr interface: "
               << dp_attr.getAbstractAttribute().hasInterface(
                      mlir::DistributeParallelAttr::getInterfaceID())
               << "\n";
  llvm::outs()
      << dp_attr.getAbstractAttribute().getName()
      << " has mlir::DataParallelAttr interface: "
      << dp_attr.hasPromiseOrImplementsInterface<mlir::DataParallelAttr>()
      << "\n";

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = getModule(builder);
  module->dump();
  module->walk([](mlir::func::FuncOp func) {
    if (auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
            func->getAttr(KDPAttrName))) {
      func->walk([&](mlir::Operation* op) {
        if (auto dis_op =
                llvm::dyn_cast_or_null<mlir::DistributeParallelOp>(op)) {
          if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
            llvm::outs() << "Apply DataParallelism to " << op->getName()
                         << "\n";
            op->erase();
          };
        }
      });
    }
  });
  module->dump();
}

void IR_Struct() {
  const char* ir =
      R"(func.func @insertion_point_outside_loop(%t : tensor<?xf32>, %sz : index,
                                        %idx : index) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %blank = tensor.empty() : tensor<5xf32>

  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    %filled = linalg.fill ins(%f : f32) outs(%blank : tensor<5xf32>) -> tensor<5xf32>

    %inserted = tensor.insert_slice %filled into %bb[%idx][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %inserted : tensor<?xf32>
  }
  return %r : tensor<?xf32>
})";
  auto context = mlir::MLIRContext();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (mlir::utils::file::ParseStr<mlir::ModuleOp>(context, module, ir).failed())
    llvm::outs() << " parse ir string failed!\n";
  auto file = std::filesystem::current_path() / "ir_struct.mlir";
  if (mlir::utils::file::PrintToFile(module.get(), file.c_str()).failed()) {
    llvm::outs() << "print module error!";
  }
}

void CH7() { IR_Struct(); }

void CH8() {  // 初始化方言注册器
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = getModule(builder);
  mlir::PassManager pm(&context);
  mlir::north_star::MarkDistributeParallelParametersPassOptions
      mark_distribute_parallel_option{.DPNums = 3, .TPNums = 1};
  pm.addPass(mlir::north_star::createMarkDistributeParallelParametersPass(
      mark_distribute_parallel_option));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::north_star::createApplyDistributeTransformPass());
  module->dump();
  if (pm.run(module).failed()) {
    llvm::outs() << "run pass error!\n";
  };
  llvm::outs() << "after pass:\n";
  module->dump();
}

void CH9() {
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  context.disableMultithreading(true);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = getModule(builder);
  mlir::PassManager pm(&context);
  mlir::north_star::MarkDistributeParallelParametersPassOptions
      mark_distribute_parallel_option{.DPNums = 3, .TPNums = 1};
  pm.addPass(mlir::north_star::createMarkDistributeParallelParametersPass(
      mark_distribute_parallel_option));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::north_star::createApplyDistributeTransformPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::north_star::createDeviceRegionFusionPass());
  module->dump();
  if (pm.run(module).failed()) {
    llvm::outs() << "run pass error!\n";
  };
  llvm::outs() << "after pass:\n";
  module->dump();
}

void CH11() {
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  context.disableMultithreading(true);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = getModule(builder);
  mlir::PassManager pm(&context);
  mlir::north_star::MarkDistributeParallelParametersPassOptions
      mark_distribute_parallel_option{.DPNums = 3, .TPNums = 1};
  pm.addPass(mlir::north_star::createMarkDistributeParallelParametersPass(
      mark_distribute_parallel_option));
  auto func_pm = pm.nest<mlir::func::FuncOp>();
  func_pm.addPass(mlir::north_star::createApplyDistributeTransformPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::north_star::createDeviceRegionFusionPass());
  module->dump();
  if (pm.run(module).failed()) {
    llvm::outs() << "run pass error!\n";
  };
  llvm::outs() << "after pass:\n";
  module->dump();
}

void CH14() {
  mlir::DialectRegistry registry;
  // 初始化上下文环境
  mlir::MLIRContext context(registry);
  context.disableMultithreading(true);
  // 加载/注册方言
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto type = mlir::north_star::NSTensorType::get(&context, {2, 2},
                                                  builder.getF32Type(), 1);
  auto const_op = builder.create<mlir::north_star::ConstOp>(
      loc, type, mlir::DenseElementsAttr::get(type, mlir::ArrayRef<float>{1.0, 2., 3., 4.}));
  const_op->dump();
}

// 1. 整体pipeline的情况以及如何一步一步lowing到LLVM IR
//     1. 有哪些Pass，这些pass的功能是什么。
//     2. 有哪些不足和不科学的地方需要注意。
//     3. 有哪些可以扩展的地方。
// 2. 如何使用 transform dialect
int main() { CH14(); }