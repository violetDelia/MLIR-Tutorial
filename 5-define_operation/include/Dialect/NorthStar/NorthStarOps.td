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


class NorthStar_Op<string mnemonic, list<Trait> traits = []> :
        Op<NorthStar_Dialect, mnemonic, traits>{
    let summary = cppNamespace#opName#" op";
    let description = [{
        $_name op;
    }];
}

class NorthStar_UnaryOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
        NorthStar_Op<mnemonic,traits#[]>{
    let arguments = !con((ins
        OperandType:$input),
        attributes);
    let results = (outs 
        ResultType:$result);
}

class NorthStar_BinaryOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
    NorthStar_Op<mnemonic,traits#[]>{
    let arguments = !con((ins
        OperandType:$lhs,
        OperandType:$rhs),
        attributes);

    let results = (outs 
        ResultType:$result);
}

class NorthStar_UnaryElementwiseOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
    NorthStar_UnaryOp<mnemonic,traits# [], OperandType, ResultType, attributes>;

class NorthStar_BinaryElementwiseOp <string mnemonic, list<Trait> traits, Type OperandType, Type ResultType = OperandType, dag attributes = (ins)>:
    NorthStar_BinaryOp<mnemonic,traits# [], OperandType, ResultType, attributes>;

def NorthStar_BufferOp : NorthStar_Op<"buffer",[]>{
    let description = [{
        generate a buffer.
    }];
    let arguments = (ins 
        Variadic<AnyNSTensor>:$tensors
       );
    let results = (outs 
     AnyBuffer:$result);
    let builders = [
        OpBuilder<(ins "::mlir::ValueRange":$tensors),
        [{
            $_state.addOperands(tensors);
            llvm::SmallVector<int64_t> divice_ids;
            for( auto tensor: tensors){
                auto tensor_type = llvm::cast_or_null<NSTensorType>(tensor.getType());
                divice_ids.push_back(tensor_type.getDeviceId());
            }
            $_state.addTypes( BufferType::get($_state.getContext(),divice_ids));
        }]>
    ];
    let hasVerifier = 1;
}

def NorthStar_GetTensorOp : NorthStar_Op<"get_tensor",[]>{
    let description = [{
        get a tensor form buffer
    }];
    let arguments = (ins 
        AnyBuffer:$buffer,
        SI64Attr:$device_id
       );
    let results = (outs 
        AnyNSTensor:$result);
    let hasVerifier = 1;
}

def NorthStar_ConstOp   : NorthStar_Op<"const", []>{
    let arguments = (ins 
        ElementsAttr:$value);
    let results = (outs
        AnyNSTensor:$result);
}

def NorthStar_SoftmaxOp   : NorthStar_UnaryElementwiseOp<"softmax", [], AnyNSTensor, AnyNSTensor, (ins I64Attr: $axis)>{
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input,"std::int64_t": $axis),
        [{
            $_state.addOperands(input);
            $_state.getOrAddProperties<Properties>().axis = $_builder.getIntegerAttr(odsBuilder.getIntegerType(64, /*isSigned=*/true), axis);;
            $_state.addTypes(input.getType());
        }]>
    ];
    let hasVerifier = 1;
}

def NorthStar_ExpOp   : NorthStar_UnaryElementwiseOp<"exp", [], AnyNSTensor>{
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input),
        [{
            $_state.addOperands(input);
            $_state.addTypes(input.getType());
        }]>
    ];
}

def NorthStar_AddOp   : NorthStar_BinaryElementwiseOp<"add", [], AnyNSTensor>;

def NorthStar_MulOp   : NorthStar_BinaryElementwiseOp<"mul", [], AnyNSTensor>;

def NorthStar_DivOp   : NorthStar_BinaryElementwiseOp<"div", [], AnyNSTensor>;

def NorthStar_SubOp   : NorthStar_BinaryElementwiseOp<"sub", [], AnyNSTensor>;


def NorthStar_ReduceSumOp   : NorthStar_Op<"reduce_sum", []>{
    let arguments = (ins 
        AnyNSTensor:$input,
        DenseI64ArrayAttr:$axis
        );
    let results = (outs 
        AnyNSTensor:$result);
}

def NorthStar_PrintOp   : NorthStar_Op<"print", []>{
    let arguments = (ins 
        AnyNSTensor:$input);
    let results = (outs);
}

def NorthStar_AllToAllOp   : NorthStar_Op<"all_to_all", []>{
    let arguments = (ins 
        AnyBuffer:$input,
        AnyBuffer:$output
        );
}

#endif // INCLUDE_LLCOMPILER_DIALECT_NorthStar_IR_LLHOPS_TD_
