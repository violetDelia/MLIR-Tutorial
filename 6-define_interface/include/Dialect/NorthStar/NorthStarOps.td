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
#ifndef DIALECT_NORTH_STAR_OPS_TD
#define DIALECT_NORTH_STAR_OPS_TD

include "Dialect/NorthStar/NorthStarAttrs.td"



class NorthStar_Op<string mnemonic, list<Trait> traits = []>
    : Op<NorthStar_Dialect, mnemonic, traits> {
  let summary = cppNamespace#opName#"op";

  let description = "$_name op";
}

class NorthStar_UnaryOp<string mnemonic,Type OperandType, Type resultType = OperandType,list<Trait> traits = [], dag attributes = (ins)>:
    NorthStar_Op<mnemonic, [DeclareOpInterfaceMethods<DistributeParallelOp>] # traits>{
    let arguments = !con((ins
        OperandType:$input),
        attributes
        );

    let results = (outs
        resultType:$result);
}

class NorthStar_BinaryOp<string mnemonic,Type OperandType, Type resultType = OperandType,list<Trait> traits = [], dag attributes = (ins)>:
    NorthStar_Op<mnemonic, [DeclareOpInterfaceMethods<DistributeParallelOp>] # traits>{
    let arguments = !con((ins
        OperandType:$lhs,
        OperandType:$rhs),
        attributes
        );

    let results = (outs
        resultType:$result);
}

class NorthStar_ElewiseUnaryOp<string mnemonic,Type OperandType, Type resultType = OperandType,list<Trait> traits = [], dag attributes = (ins)>:
    NorthStar_UnaryOp<mnemonic, OperandType, resultType, [DeclareOpInterfaceMethods<SupportedDataParallelismOp>] # traits, attributes>;

class NorthStar_ElewiseBinaryOp<string mnemonic,Type OperandType, Type resultType = OperandType,list<Trait> traits = [], dag attributes = (ins)>:
    NorthStar_BinaryOp<mnemonic, OperandType, resultType, [DeclareOpInterfaceMethods<SupportedDataParallelismOp>] # traits,attributes>;


def NorthStar_ConstOp : NorthStar_Op<"const",[]>{
    let arguments = (ins
        ElementsAttr:$value);
    let results = (outs
        AnyNSTensor:$result);
}

def NorthStar_SoftmaxOp : NorthStar_UnaryOp<"softmax",AnyNSTensor, AnyNSTensor, [DeclareOpInterfaceMethods<SupportedDataParallelismOp,["applyDataParallelism","supportedDataParallelism"]>], (ins I64Attr:$axis)>{
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis) , 
            [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis = $_builder.getIntegerAttr(odsBuilder.getIntegerType(64,true), axis);
                $_state.addTypes(input.getType());
            }]>

    ];
}


def NorthStar_ExpOp : NorthStar_ElewiseUnaryOp<"exp",AnyNSTensor>{
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input) , 
            [{
                $_state.addOperands(input);
                $_state.addTypes(input.getType());
            }]>
    ];
}


def NorthStar_AddOp : NorthStar_ElewiseBinaryOp<"add",AnyNSTensor>;
def NorthStar_SubOp : NorthStar_ElewiseBinaryOp<"sub",AnyNSTensor>;
def NorthStar_MulOp : NorthStar_ElewiseBinaryOp<"mul",AnyNSTensor>;
def NorthStar_DivOp : NorthStar_ElewiseBinaryOp<"div",AnyNSTensor>;


def NorthStar_AllToAllOp : NorthStar_Op<"all_to_all",[]>{
    let arguments = (ins
        AnyBuffer:$input,
        AnyBuffer:$output
    );
}


def NorthStar_BufferCastOp : NorthStar_Op<"buffer_cast",[]>{
    let description = "对数据切分的标记";
    let arguments = (ins
        Variadic<NSTensorOrBuffer>:$inputs,
        DistributeParallelAttr:$distribute_attr
    );
    let results = (outs
        Variadic<NSTensorOrBuffer>:$outputs);
    let hasVerifier = 1;
}

def NorthStar_BufferOp : NorthStar_Op<"buffer",[]>{
    let description = "将 多个device_id 的tensor 组合成 一个 buffer";
    let arguments = (ins
        Variadic<AnyNSTensor>:$tensors
    );
    let results = (outs
        AnyBuffer:$result);
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::ValueRange":$tensors) , 
            [{
                $_state.addOperands(tensors);
                ::llvm::SmallVector<int64_t> devices;
                for (auto tensor : tensors) {
                    auto tensor_type =
                        llvm::cast<::mlir::north_star::NSTensorType>(tensor.getType());
                    devices.push_back(tensor_type.getDeviceId());
                }
                $_state.addTypes(::mlir::north_star::BufferType::get($_state.getContext(), devices));
            }]>
    ];
}

def NorthStar_GetTensorOp: NorthStar_Op<"get_tensor",[]>{
    let description = "从buffer中取出指定device_id的tensor";
    let arguments = (ins
        AnyBuffer:$buffer,
        I64Attr:$device_id
    );
    let results = (outs
        AnyNSTensor:$result);
    let hasVerifier = 1;
}

def NorthStar_PrintOp: NorthStar_Op<"print",[]>{
    let arguments = (ins
        AnyType:$input
    );
}


#endif // INCLUDE_LLCOMPILER_DIALECT_NorthStar_IR_LLHOPS_TD_
