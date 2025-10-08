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

#include "Conversion/NorthStarToFunc/NorthStarToFunc.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "Dialect/NorthStar/IR/NorthStarOps.h"
#include "Dialect/NorthStar/IR/NorthStarTypes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/FuncBuilder.h"
#include "Utils/Key.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
namespace {

struct TensorToNSTensorOpConversionPattern final
    : public OpConversionPattern<mlir::north_star::TensorToNSTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::TensorToNSTensorOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::TensorToNSTensorOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto context = op.getContext();
    auto loc = op->getLoc();
    llvm::SmallVector<Type> converted_operand_types;
    if (getTypeConverter()
            ->convertTypes(op->getOperands().getTypes(),
                           converted_operand_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    llvm::SmallVector<Type> converted_result_types;
    if (getTypeConverter()
            ->convertTypes(op->getResultTypes(), converted_result_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    llvm::SmallVector<Type> new_func_input_types;
    new_func_input_types.push_back(rewriter.getI64Type());

    new_func_input_types.push_back(
        getTypeConverter()->convertType(op.getInput().getType()));
    auto device_id =
        rewriter.create<arith::ConstantIntOp>(loc, op.getDeviceId(), 64);
    mlir::utils::AutoCastOption options;
    options.castTensor = true;
    utils::FunctionCallBuilder builder(
        op.getBuildinFunctionName(),
        FunctionType::get(
            context, new_func_input_types,
            getTypeConverter()->convertType(op.getResult().getType())),
        options);
    auto builder_res = builder.create(
        op->getLoc(), rewriter, ValueRange{device_id, adaptor.getInput()});

    rewriter.replaceOp(op, builder_res.replecement_results);
  }
};

struct NSTensorToTensorOpConversionPattern final
    : public OpConversionPattern<mlir::north_star::NSTensorToTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::NSTensorToTensorOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::NSTensorToTensorOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto context = op.getContext();
    auto loc = op->getLoc();
    llvm::SmallVector<Type> converted_operand_types;
    if (getTypeConverter()
            ->convertTypes(op->getOperands().getTypes(),
                           converted_operand_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    llvm::SmallVector<Type> converted_result_types;
    if (getTypeConverter()
            ->convertTypes(op->getResultTypes(), converted_result_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    llvm::SmallVector<Type> new_func_input_types;
    new_func_input_types.push_back(
        getTypeConverter()->convertType(op.getInput().getType()));
    mlir::utils::AutoCastOption options;
    options.castTensor = true;
    utils::FunctionCallBuilder builder(
        op.getBuildinFunctionName(),
        FunctionType::get(
            context, new_func_input_types,
            getTypeConverter()->convertType(op.getResult().getType())),
        options);
    auto builder_res =
        builder.create(op->getLoc(), rewriter, ValueRange{adaptor.getInput()});

    rewriter.replaceOp(op, builder_res.replecement_results);
  }
};

struct BufferOpConversionPattern final
    : public OpConversionPattern<mlir::north_star::BufferOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::BufferOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::BufferOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto context = op.getContext();
    auto loc = op->getLoc();
    llvm::SmallVector<Type> converted_operand_types;
    if (getTypeConverter()
            ->convertTypes(op.getOperandTypes(), converted_operand_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    llvm::SmallVector<Type> converted_result_types;
    if (getTypeConverter()
            ->convertTypes(op->getResultTypes(), converted_result_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    auto converted_type = converted_operand_types[0];
    auto ptr_type = LLVM::LLVMPointerType::get(context);
    Value device_nums = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(64), op->getNumOperands());
    auto memrefs = rewriter.create<LLVM::AllocaOp>(loc, ptr_type,
                                                   converted_type, device_nums,
                                                   /*alignment=*/0);
    auto device_indexs = rewriter.create<LLVM::AllocaOp>(
        loc, ptr_type, rewriter.getI64Type(), device_nums,
        /*alignment=*/0);
    auto device_ids =
        llvm::cast<north_star::BufferType>(op.getResult().getType())
            .getDevices();
    for (auto [index, operand, device_id] :
         llvm::enumerate(adaptor.getOperands(), device_ids)) {
      auto ptr =
          rewriter.create<LLVM::GEPOp>(loc, ptr_type, converted_type, memrefs,
                                       ArrayRef<LLVM::GEPArg>{(int32_t)index});
      rewriter.create<LLVM::StoreOp>(loc, operand, ptr);
      auto index_ptr = rewriter.create<LLVM::GEPOp>(
          loc, ptr_type, rewriter.getI64Type(), device_indexs,
          ArrayRef<LLVM::GEPArg>{(int32_t)index});
      auto index_val =
          rewriter.create<arith::ConstantIntOp>(loc, device_id, 64);
      rewriter.create<LLVM::StoreOp>(loc, index_val, index_ptr);
    }
    mlir::utils::AutoCastOption options;
    options.castTensor = true;
    utils::FunctionCallBuilder builder(
        op.getBuildinFunctionName(),
        FunctionType::get(context,
                          TypeRange{ptr_type, ptr_type, rewriter.getI64Type()},
                          converted_result_types),
        options);
    auto builder_res =
        builder.create(op->getLoc(), rewriter,
                       ValueRange{memrefs, device_indexs, device_nums});
    rewriter.replaceOp(op, builder_res.replecement_results);
  }
};

struct GetTensorOpConversionPattern final
    : public OpConversionPattern<mlir::north_star::GetTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::GetTensorOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::GetTensorOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto context = op.getContext();
    auto loc = op->getLoc();
    llvm::SmallVector<Type> converted_operand_types;
    if (getTypeConverter()
            ->convertTypes(op->getOperandTypes(), converted_operand_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    llvm::SmallVector<Type> converted_result_types;
    if (getTypeConverter()
            ->convertTypes(op->getResultTypes(), converted_result_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    auto device_index =
        rewriter.create<arith::ConstantIntOp>(loc, op.getDeviceId(), 64);
    mlir::utils::AutoCastOption options;
    options.castTensor = true;
    utils::FunctionCallBuilder builder(
        op.getBuildinFunctionName(),
        FunctionType::get(
            context,
            TypeRange{rewriter.getI64Type(), getTypeConverter()->convertType(
                                                 op.getBuffer().getType())},
            converted_result_types),
        options);
    auto builder_res = builder.create(
        op->getLoc(), rewriter, ValueRange{device_index, adaptor.getBuffer()});
    rewriter.replaceOp(op, builder_res.replecement_results);
  }
};

struct GatherOpOpConversionPattern final
    : public OpConversionPattern<mlir::north_star::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::GatherOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::GatherOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto context = op.getContext();
    auto loc = op->getLoc();
    llvm::SmallVector<Type> converted_operand_types;
    if (getTypeConverter()
            ->convertTypes(op.getOperandTypes(), converted_operand_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    llvm::SmallVector<Type> converted_result_types;
    if (getTypeConverter()
            ->convertTypes(op->getResultTypes(), converted_result_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    mlir::utils::AutoCastOption options;
    options.castTensor = true;
    utils::FunctionCallBuilder builder(
        op.getBuildinFunctionName(),
        FunctionType::get(context, converted_operand_types,
                          converted_result_types),
        options);
    auto builder_res =
        builder.create(op->getLoc(), rewriter, adaptor.getOperands());
    rewriter.replaceOp(op, builder_res.replecement_results);
  }
};

struct ScatterOpOpConversionPattern final
    : public OpConversionPattern<mlir::north_star::ScatterOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult match(north_star::ScatterOp op) const final {
    return llvm::success();
  }
  void rewrite(north_star::ScatterOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto context = op.getContext();
    auto loc = op->getLoc();
    llvm::SmallVector<Type> converted_operand_types;
    if (getTypeConverter()
            ->convertTypes(op.getOperandTypes(), converted_operand_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    llvm::SmallVector<Type> converted_result_types;
    if (getTypeConverter()
            ->convertTypes(op->getResultTypes(), converted_result_types)
            .failed()) {
      op.emitError() << "type convert failed";
      return;
    }
    mlir::utils::AutoCastOption options;
    options.castTensor = true;
    utils::FunctionCallBuilder builder(
        op.getBuildinFunctionName(),
        FunctionType::get(context, converted_operand_types,
                          converted_result_types),
        options);
    auto builder_res =
        builder.create(op->getLoc(), rewriter, adaptor.getOperands());
    rewriter.replaceOp(op, builder_res.replecement_results);
  }
};

}  // namespace

namespace mlir::north_star {
namespace {}  // namespace
void initNorthStarToFuncTypeConvert(TypeConverter &type_converter) {
  type_converter.addConversion([](TensorType type) { return type; });
  type_converter.addConversion([](NSTensorType type) {
    auto context = type.getContext();
    llvm::SmallVector<Type> types;
    auto I64 = IntegerType::get(context, 64);
    auto ptr_type = LLVM::LLVMPointerType::get(context);
    types.push_back(I64);
    SmallVector<Type, 2> unranded_memref_types = {I64, ptr_type};
    auto unranded_memref_struct_type =
        LLVM::LLVMStructType::getLiteral(context, unranded_memref_types);
    types.push_back(unranded_memref_struct_type);
    return LLVM::LLVMStructType::getLiteral(context, types);
  });
  type_converter.addConversion([](BufferType type) {
    auto context = type.getContext();

    auto I64 = IntegerType::get(context, 64);
    auto ptr_type = LLVM::LLVMPointerType::get(context);
    llvm::SmallVector<Type> types = {I64, ptr_type, ptr_type, ptr_type};
    return LLVM::LLVMStructType::getLiteral(context, types);
  });

  auto materialize_cast = [](OpBuilder &builder, Type type, ValueRange inputs,
                             Location loc) -> std::optional<Value> {
    if (inputs.size() != 1) return std::nullopt;
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
        .getResult(0);
  };

  type_converter.addSourceMaterialization(materialize_cast);
  type_converter.addTargetMaterialization(materialize_cast);
  type_converter.addArgumentMaterialization(materialize_cast);
}

void populateNorthStarToFuncPatterns(TypeConverter &type_converter,
                                     RewritePatternSet &patterns) {
  patterns.add<TensorToNSTensorOpConversionPattern,
               NSTensorToTensorOpConversionPattern, BufferOpConversionPattern,
               GetTensorOpConversionPattern, ScatterOpOpConversionPattern,
               GatherOpOpConversionPattern>(type_converter,
                                            patterns.getContext());
};
}  // namespace mlir::north_star
